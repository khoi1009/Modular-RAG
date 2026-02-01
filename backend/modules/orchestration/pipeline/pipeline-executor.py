"""Pipeline executor for orchestrated query processing."""
import asyncio
import time
from typing import Any, Dict

import async_timeout

from backend.logger import logger
from backend.modules.orchestration.pipeline.condition_evaluator import (
    ConditionEvaluator,
)
from backend.modules.orchestration.pipeline.schemas import PipelineDefinition
from backend.modules.orchestration.pipeline.step_registry import StepRegistry
from backend.modules.orchestration.schemas import PipelineResult


class PipelineExecutor:
    """
    Executes pipeline definitions with conditional steps.
    """

    def __init__(
        self,
        step_registry: StepRegistry = None,
        condition_evaluator: ConditionEvaluator = None,
    ):
        """
        Initialize pipeline executor.

        Args:
            step_registry: Registry of step functions
            condition_evaluator: Evaluator for step conditions
        """
        self.registry = step_registry or StepRegistry()
        self.evaluator = condition_evaluator or ConditionEvaluator()

    async def execute(
        self, pipeline_def: PipelineDefinition, initial_context: Dict[str, Any]
    ) -> PipelineResult:
        """
        Execute a pipeline definition.

        Args:
            pipeline_def: Pipeline definition with steps
            initial_context: Initial context dictionary

        Returns:
            PipelineResult with execution details
        """
        context = {**initial_context, **pipeline_def.default_config}
        steps_executed = []
        errors = []
        start_time = time.time()

        logger.info(f"Starting pipeline execution: {pipeline_def.name}")

        for step in pipeline_def.steps:
            try:
                # Check condition
                if step.condition:
                    should_execute = self.evaluator.evaluate(step.condition, context)
                    if not should_execute:
                        logger.debug(
                            f"Skipping step '{step.name}' - condition not met: {step.condition}"
                        )
                        continue

                # Execute step
                logger.debug(f"Executing step: {step.name}")
                result = await self._execute_step(step, context)

                # Store result in context
                context[step.output] = result
                steps_executed.append(step.name)
                logger.debug(f"Step '{step.name}' completed successfully")

            except asyncio.TimeoutError:
                error_msg = f"Step '{step.name}' timed out after {step.timeout_sec}s"
                logger.error(error_msg)
                errors.append(error_msg)
                # Continue to next step on timeout
                continue

            except Exception as e:
                error_msg = f"Step '{step.name}' failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

                # Retry logic
                if step.retry_count > 0:
                    logger.info(f"Retrying step '{step.name}'...")
                    for retry in range(step.retry_count):
                        try:
                            await asyncio.sleep(0.5 * (retry + 1))  # Exponential backoff
                            result = await self._execute_step(step, context)
                            context[step.output] = result
                            steps_executed.append(f"{step.name}_retry_{retry + 1}")
                            logger.info(f"Step '{step.name}' succeeded on retry {retry + 1}")
                            break
                        except Exception as retry_error:
                            logger.error(f"Retry {retry + 1} failed: {retry_error}")
                            if retry == step.retry_count - 1:
                                errors.append(f"All retries failed for '{step.name}'")

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Determine success
        success = len(errors) == 0 and len(steps_executed) > 0

        result = PipelineResult(
            success=success,
            answer=context.get("answer"),
            sources=context.get("sources", []),
            context=context,
            execution_time_ms=execution_time_ms,
            steps_executed=steps_executed,
            errors=errors,
        )

        logger.info(
            f"Pipeline '{pipeline_def.name}' completed: "
            f"success={success}, steps={len(steps_executed)}, time={execution_time_ms}ms"
        )

        return result

    async def _execute_step(self, step, context: Dict[str, Any]) -> Any:
        """
        Execute a single pipeline step with timeout.

        Args:
            step: PipelineStep definition
            context: Current context

        Returns:
            Step result

        Raises:
            asyncio.TimeoutError: If step exceeds timeout
            ValueError: If step module not found
        """
        # Get step function
        step_fn = self.registry.get(step.module)

        # Prepare input
        if step.input:
            input_data = context.get(step.input)
            if input_data is None:
                logger.warning(
                    f"Input key '{step.input}' not found in context for step '{step.name}'"
                )
        else:
            input_data = context

        # Execute with timeout
        async with async_timeout.timeout(step.timeout_sec):
            if step.parallel:
                # For parallel execution, expect input to be iterable
                result = await self._execute_parallel(step_fn, input_data)
            else:
                # Standard sequential execution
                if asyncio.iscoroutinefunction(step_fn):
                    result = await step_fn(input_data)
                else:
                    # If function is not async, run in executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, step_fn, input_data)

        return result

    async def _execute_parallel(self, step_fn, input_data):
        """
        Execute step function in parallel over input items.

        Args:
            step_fn: Step function to execute
            input_data: List or iterable of inputs

        Returns:
            List of results
        """
        if not isinstance(input_data, (list, tuple)):
            logger.warning("Parallel execution expects list/tuple input, executing single item")
            if asyncio.iscoroutinefunction(step_fn):
                return await step_fn(input_data)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, step_fn, input_data)

        # Execute all items in parallel
        tasks = []
        for item in input_data:
            if asyncio.iscoroutinefunction(step_fn):
                tasks.append(step_fn(item))
            else:
                loop = asyncio.get_event_loop()
                tasks.append(loop.run_in_executor(None, step_fn, item))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Parallel task {i} failed: {result}")
            else:
                valid_results.append(result)

        return valid_results
