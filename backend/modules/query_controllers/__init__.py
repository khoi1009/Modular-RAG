from backend.modules.query_controllers.example.controller import BasicRAGQueryController
from backend.modules.query_controllers.multimodal.controller import (
    MultiModalRAGQueryController,
)
from backend.modules.query_controllers.orchestrated.orchestrated_rag_query_controller import (
    OrchestratedQueryController,
)
from backend.modules.query_controllers.water_infrastructure.water_infrastructure_controller import (
    WaterInfrastructureController,
)
from backend.modules.query_controllers.query_controller import register_query_controller

register_query_controller("basic-rag", BasicRAGQueryController)
register_query_controller("multimodal", MultiModalRAGQueryController)
register_query_controller("orchestrated", OrchestratedQueryController)
register_query_controller("water-infrastructure", WaterInfrastructureController)
