"""Temporal expression parser for water infrastructure queries."""
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple


class TemporalParser:
    """Parse natural language date/time expressions into date ranges."""

    def parse(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Parse date range from query text.

        Supports expressions like:
        - "last 7 days"
        - "past week"
        - "Q1 2024"
        - "January 2024"
        - "yesterday"
        - "this month"
        """
        query_lower = query.lower()

        # Try relative date patterns first
        date_range = self._parse_relative_dates(query_lower)
        if date_range:
            return date_range

        # Try quarter patterns
        date_range = self._parse_quarters(query_lower)
        if date_range:
            return date_range

        # Try month/year patterns
        date_range = self._parse_month_year(query_lower)
        if date_range:
            return date_range

        return None

    def _parse_relative_dates(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """Parse relative date expressions."""
        now = datetime.now()

        # Last N days/weeks/months
        patterns = {
            r'last\s+(\d+)\s+days?': lambda n: (now - timedelta(days=int(n)), now),
            r'past\s+(\d+)\s+days?': lambda n: (now - timedelta(days=int(n)), now),
            r'last\s+(\d+)\s+weeks?': lambda n: (now - timedelta(weeks=int(n)), now),
            r'past\s+(\d+)\s+weeks?': lambda n: (now - timedelta(weeks=int(n)), now),
            r'last\s+(\d+)\s+months?': lambda n: (self._subtract_months(now, int(n)), now),
            r'past\s+(\d+)\s+months?': lambda n: (self._subtract_months(now, int(n)), now),
        }

        for pattern, calc_func in patterns.items():
            match = re.search(pattern, query)
            if match:
                return calc_func(match.group(1))

        # Named periods
        named_periods = {
            'yesterday': (
                now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1),
                now.replace(hour=23, minute=59, second=59, microsecond=999999) - timedelta(days=1),
            ),
            'today': (
                now.replace(hour=0, minute=0, second=0, microsecond=0),
                now,
            ),
            'this week': (
                now - timedelta(days=now.weekday()),
                now,
            ),
            'last week': (
                now - timedelta(days=now.weekday() + 7),
                now - timedelta(days=now.weekday() + 1),
            ),
            'this month': (
                now.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
                now,
            ),
            'last month': self._get_last_month_range(now),
            'this year': (
                now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0),
                now,
            ),
        }

        for period, date_range in named_periods.items():
            if period in query:
                return date_range

        return None

    def _parse_quarters(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """Parse quarter expressions like 'Q1 2024'."""
        pattern = r'q([1-4])\s+(\d{4})'
        match = re.search(pattern, query)

        if match:
            quarter = int(match.group(1))
            year = int(match.group(2))

            quarter_months = {
                1: (1, 3),
                2: (4, 6),
                3: (7, 9),
                4: (10, 12),
            }

            start_month, end_month = quarter_months[quarter]
            start_date = datetime(year, start_month, 1)

            # Get last day of end month
            if end_month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year, end_month + 1, 1) - timedelta(days=1)

            end_date = end_date.replace(hour=23, minute=59, second=59)

            return (start_date, end_date)

        return None

    def _parse_month_year(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """Parse month/year expressions like 'January 2024'."""
        months = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12,
        }

        for month_name, month_num in months.items():
            pattern = rf'{month_name}\s+(\d{{4}})'
            match = re.search(pattern, query)
            if match:
                year = int(match.group(1))
                start_date = datetime(year, month_num, 1)

                # Get last day of month
                if month_num == 12:
                    end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    end_date = datetime(year, month_num + 1, 1) - timedelta(days=1)

                end_date = end_date.replace(hour=23, minute=59, second=59)

                return (start_date, end_date)

        return None

    def _subtract_months(self, dt: datetime, months: int) -> datetime:
        """Subtract months from a datetime."""
        month = dt.month - months
        year = dt.year

        while month <= 0:
            month += 12
            year -= 1

        return dt.replace(year=year, month=month)

    def _get_last_month_range(self, now: datetime) -> Tuple[datetime, datetime]:
        """Get date range for last month."""
        if now.month == 1:
            last_month = 12
            year = now.year - 1
        else:
            last_month = now.month - 1
            year = now.year

        start_date = datetime(year, last_month, 1, 0, 0, 0)

        # Get last day of that month
        if last_month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, last_month + 1, 1) - timedelta(days=1)

        end_date = end_date.replace(hour=23, minute=59, second=59)

        return (start_date, end_date)
