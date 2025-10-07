#!/usr/bin/env python3
"""
Model Performance Benchmark Script

This script benchmarks different LLM models for latency and performance
in the context of Vectara Agentic framework.
"""

import asyncio
import time
import json
import statistics
import sys
import os
import random
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass, asdict

# Add the current directory to Python path to import vectara_agentic
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vectara_agentic.agent import Agent
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.types import ModelProvider, ObserverType
from vectara_agentic.tools import ToolsFactory
from vectara_agentic._observability import setup_observer, shutdown_observer

# Initialize observability once at startup to prevent repeated instrumentation
_observability_initialized = False


def validate_api_keys(models_to_test: List[Dict]) -> None:
    """
    Validate that all required API keys are present for the models being tested.

    Args:
        models_to_test: List of model configurations with provider and model info

    Raises:
        SystemExit: If any required API keys are missing
    """
    # Map providers to their required environment variables
    provider_api_keys = {
        ModelProvider.OPENAI: "OPENAI_API_KEY",
        ModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        ModelProvider.TOGETHER: "TOGETHER_API_KEY",
        ModelProvider.GROQ: "GROQ_API_KEY",
        ModelProvider.GEMINI: "GOOGLE_API_KEY",
    }

    required_keys = set()

    # Collect unique providers from models to test
    providers_in_use: Set[ModelProvider] = set()
    for model_config in models_to_test:
        providers_in_use.add(model_config["provider"])

    # Add required API keys for each provider
    for provider in providers_in_use:
        api_key_name = provider_api_keys.get(provider)
        if api_key_name:  # Skip providers that don't use env var API keys
            required_keys.add(api_key_name)

    # Check for missing API keys
    missing_keys = []
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)

    if missing_keys:
        print("ERROR: Missing required API keys for benchmark execution:")
        print()
        for key in sorted(missing_keys):
            print(f"  • {key}")
        print()
        print("Please set these environment variables before running the benchmark.")
        print("Providers being tested:")
        for provider in sorted(providers_in_use, key=lambda p: p.value):
            models_for_provider = [
                m["model"] for m in models_to_test if m["provider"] == provider
            ]
            print(f"  • {provider.value}: {', '.join(models_for_provider)}")

        sys.exit(1)

    print("All required API keys are present")
    print(f"Found API keys for {len(required_keys)} required environment variables")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    model_name: str
    provider: str
    test_type: str
    first_token_latency: float
    total_response_time: float
    response_length: int
    tokens_per_second: float
    error: str = None


@dataclass
class BenchmarkStats:
    """Aggregated statistics for multiple runs."""

    model_name: str
    provider: str
    test_type: str
    runs: int
    avg_first_token_latency: float
    avg_total_response_time: float
    avg_tokens_per_second: float
    median_first_token_latency: float
    median_total_response_time: float
    median_tokens_per_second: float
    min_total_response_time: float
    max_total_response_time: float
    std_total_response_time: float
    success_rate: float


class ModelBenchmark:
    """Benchmarking suite for different LLM models."""

    def __init__(
        self, enable_observability: bool = False, max_concurrent_models: int = 2
    ):
        # Test configurations
        self.enable_observability = enable_observability
        self.max_concurrent_models = max_concurrent_models
        self.models_to_test = [
            # OpenAI models
            {"provider": ModelProvider.OPENAI, "model": "gpt-5-mini"},
            {"provider": ModelProvider.OPENAI, "model": "gpt-4o-mini"},
            {"provider": ModelProvider.OPENAI, "model": "gpt-4.1-mini"},
            {"provider": ModelProvider.ANTHROPIC, "model": "claude-sonnet-4-5"},
            {"provider": ModelProvider.TOGETHER, "model": "deepseek-ai/DeepSeek-V3"},
            {"provider": ModelProvider.GROQ, "model": "openai/gpt-oss-20b"},
            {"provider": ModelProvider.GEMINI, "model": "models/gemini-2.5-flash-lite"},
        ]

        # Test scenarios - focused on advanced tool calling only
        self.test_scenarios = {
            "financial_analysis": {
                "prompt": "Analyze a $50,000 investment portfolio with 60% stocks (8% return), 30% bonds (4% return), and 10% cash (1% return). Calculate the expected annual return, then determine how the portfolio value would grow over 15 years with monthly contributions of $1,000. Create a summary report of the analysis.",
                "description": "Multi-step financial analysis with calculations and reporting",
                "needs_tools": True,
            },
            "data_processing": {
                "prompt": "Generate a dataset of 100 customers with randomized demographics (age, income, location, purchase_history). Then analyze this data to find correlations between age groups and spending patterns. Create a statistical summary and export the results to a formatted report.",
                "description": "Data generation, analysis, and reporting workflow",
                "needs_tools": True,
            },
            "research_synthesis": {
                "prompt": "Search for information about the latest developments in quantum computing, specifically focusing on error correction breakthroughs in 2024. Extract key findings from multiple sources, summarize the technical approaches, and create a structured research report with citations.",
                "description": "Information retrieval, synthesis, and document generation",
                "needs_tools": True,
            },
            "system_monitoring": {
                "prompt": "Check system performance metrics including CPU usage, memory consumption, and disk space. If any metrics exceed safe thresholds (CPU > 80%, Memory > 90%, Disk > 85%), generate alerts and suggest optimization strategies. Create a monitoring report with recommendations.",
                "description": "System monitoring with conditional logic and reporting",
                "needs_tools": True,
            },
            "workflow_automation": {
                "prompt": "Create a project task list with 10 software development tasks, assign priorities and estimated hours, then simulate a sprint planning session by organizing tasks into a 2-week sprint. Generate a sprint backlog with daily breakdowns and resource allocation recommendations.",
                "description": "Complex workflow orchestration with multiple tool interactions",
                "needs_tools": True,
            },
        }

        self.iterations_per_test = 5
        self.results: List[BenchmarkResult] = []

        # Provider-specific rate limits (requests per minute)
        self.provider_rate_limits = {
            ModelProvider.OPENAI: 100,
            ModelProvider.ANTHROPIC: 100,
            ModelProvider.TOGETHER: 80,
            ModelProvider.GROQ: 50,  # Conservative for GROQ
            ModelProvider.GEMINI: 60,
        }

    def create_agent_config(
        self, provider: ModelProvider, model_name: str
    ) -> AgentConfig:
        """Create agent configuration for the specified model."""
        return AgentConfig(
            main_llm_provider=provider,
            main_llm_model_name=model_name,
            tool_llm_provider=provider,
            tool_llm_model_name=model_name,
            observer=(
                ObserverType.ARIZE_PHOENIX
                if self.enable_observability
                else ObserverType.NO_OBSERVER
            ),
        )

    def analyze_customer_data(self, customer_data_json: str) -> dict:
        """Analyze customer data for patterns and correlations."""
        customers = json.loads(customer_data_json)

        # Group by age groups
        age_groups = {}
        for customer in customers:
            group = customer["age_group"]
            if group not in age_groups:
                age_groups[group] = {
                    "count": 0,
                    "total_spending": 0,
                    "total_income": 0,
                }

            age_groups[group]["count"] += 1
            age_groups[group]["total_spending"] += customer["purchase_history"]
            age_groups[group]["total_income"] += customer["income"]

        # Calculate averages
        analysis = {}
        for group, data in age_groups.items():
            analysis[group] = {
                "count": data["count"],
                "avg_spending": round(data["total_spending"] / data["count"], 2),
                "avg_income": round(data["total_income"] / data["count"], 2),
                "spending_to_income_ratio": round(
                    (data["total_spending"] / data["count"])
                    / (data["total_income"] / data["count"])
                    * 1000,
                    4,
                ),
            }

        return {
            "total_customers": len(customers),
            "age_group_analysis": analysis,
            "overall_avg_spending": round(
                sum(c["purchase_history"] for c in customers) / len(customers), 2
            ),
            "overall_avg_income": round(
                sum(c["income"] for c in customers) / len(customers), 2
            ),
        }

    def get_system_metrics(self) -> dict:
        """Get current system performance metrics."""
        import psutil
        from datetime import datetime

        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception:
            # Fallback with simulated data for testing
            return {
                "cpu_usage_percent": random.randint(20, 95),
                "memory_usage_percent": random.randint(40, 95),
                "memory_available_gb": random.randint(1, 16),
                "disk_usage_percent": random.randint(30, 90),
                "disk_free_gb": random.randint(10, 500),
                "timestamp": datetime.now().isoformat(),
                "note": "Simulated data - psutil unavailable",
            }

    def check_system_health(
        self,
        cpu_threshold: int = 80,
        memory_threshold: int = 90,
        disk_threshold: int = 85,
    ) -> dict:
        """Check system health against thresholds and generate alerts."""
        metrics = self.get_system_metrics()
        alerts = []
        recommendations = []

        if metrics["cpu_usage_percent"] > cpu_threshold:
            alerts.append(
                f"HIGH CPU USAGE: {metrics['cpu_usage_percent']}% (threshold: {cpu_threshold}%)"
            )
            recommendations.append(
                "Consider closing unnecessary applications or upgrading CPU"
            )

        if metrics["memory_usage_percent"] > memory_threshold:
            alerts.append(
                f"HIGH MEMORY USAGE: {metrics['memory_usage_percent']}% (threshold: {memory_threshold}%)"
            )
            recommendations.append(
                "Close memory-intensive applications or add more RAM"
            )

        if metrics["disk_usage_percent"] > disk_threshold:
            alerts.append(
                f"LOW DISK SPACE: {metrics['disk_usage_percent']}% used (threshold: {disk_threshold}%)"
            )
            recommendations.append("Clean up temporary files or expand disk storage")

        health_status = (
            "CRITICAL" if len(alerts) >= 2 else "WARNING" if alerts else "HEALTHY"
        )

        return {
            "health_status": health_status,
            "alerts": alerts,
            "recommendations": recommendations,
            "metrics": metrics,
            "thresholds": {
                "cpu": cpu_threshold,
                "memory": memory_threshold,
                "disk": disk_threshold,
            },
        }

    def create_project_tasks(self, count: int = 10) -> str:
        """Generate a list of software development tasks."""
        task_types = [
            "Implement user authentication system",
            "Create REST API endpoints",
            "Design database schema",
            "Build responsive frontend components",
            "Write unit tests",
            "Set up CI/CD pipeline",
            "Implement error handling",
            "Create API documentation",
            "Optimize database queries",
            "Implement caching layer",
            "Add logging and monitoring",
            "Create user dashboard",
            "Implement search functionality",
            "Add data validation",
            "Create admin panel",
        ]

        tasks = []
        for i in range(count):
            task = random.choice(task_types)
            priority = random.choice(["High", "Medium", "Low"])
            estimated_hours = random.randint(2, 24)

            tasks.append(
                {
                    "task_id": f"TASK-{i+1:03d}",
                    "title": f"{task} #{i+1}",
                    "priority": priority,
                    "estimated_hours": estimated_hours,
                    "status": "Backlog",
                    "assigned_to": None,
                }
            )

        return json.dumps(tasks, indent=2)

    def plan_sprint(self, tasks_json: str, sprint_capacity_hours: int = 80) -> dict:
        """Organize tasks into a sprint with daily breakdowns."""
        tasks = json.loads(tasks_json)

        # Sort by priority and estimated hours
        priority_order = {"High": 3, "Medium": 2, "Low": 1}
        tasks.sort(
            key=lambda x: (priority_order[x["priority"]], -x["estimated_hours"]),
            reverse=True,
        )

        sprint_tasks = []
        total_hours = 0

        for task in tasks:
            if total_hours + task["estimated_hours"] <= sprint_capacity_hours:
                sprint_tasks.append(task)
                total_hours += task["estimated_hours"]
            else:
                break

        # Distribute across 2 weeks (10 working days)
        daily_breakdown = []
        remaining_hours = total_hours
        days_remaining = 10

        for day in range(1, 11):
            if days_remaining > 0:
                day_hours = min(
                    8,
                    remaining_hours // days_remaining
                    + (1 if remaining_hours % days_remaining else 0),
                )
                daily_breakdown.append(
                    {
                        "day": day,
                        "planned_hours": day_hours,
                        "remaining_capacity": 8 - day_hours,
                    }
                )
                remaining_hours -= day_hours
                days_remaining -= 1

        return {
            "sprint_summary": {
                "total_tasks": len(sprint_tasks),
                "total_planned_hours": total_hours,
                "sprint_capacity": sprint_capacity_hours,
                "utilization_percent": round(
                    (total_hours / sprint_capacity_hours) * 100, 1
                ),
            },
            "selected_tasks": sprint_tasks,
            "daily_breakdown": daily_breakdown,
            "backlog_remaining": len(tasks) - len(sprint_tasks),
        }

    def create_formatted_report(
        self, title: str, data: dict, report_type: str = "summary"
    ) -> str:
        """Create a formatted text report from structured data."""
        from datetime import datetime

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"{title.upper()}")
        report_lines.append("=" * 60)
        report_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(f"Report Type: {report_type.title()}")
        report_lines.append("")

        def format_dict(d, indent=0):
            lines = []
            for key, value in d.items():
                prefix = "  " * indent
                if isinstance(value, dict):
                    lines.append(f"{prefix}{key.replace('_', ' ').title()}:")
                    lines.extend(format_dict(value, indent + 1))
                elif isinstance(value, list):
                    lines.append(f"{prefix}{key.replace('_', ' ').title()}:")
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            lines.append(f"{prefix}  Item {i+1}:")
                            lines.extend(format_dict(item, indent + 2))
                        else:
                            lines.append(f"{prefix}  - {item}")
                else:
                    lines.append(f"{prefix}{key.replace('_', ' ').title()}: {value}")
            return lines

        report_lines.extend(format_dict(data))
        report_lines.append("")
        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    def search_information(self, query: str, max_results: int = 5) -> dict:
        """Simulate information search with structured results."""
        from datetime import datetime

        # Simulated search results for testing
        simulated_results = [
            {
                "title": f"Research Paper: {query} - Latest Developments",
                "source": "Journal of Advanced Computing",
                "summary": f"Recent breakthrough in {query} showing promising results in error reduction and scalability improvements.",
                "relevance_score": random.randint(80, 95),
                "publication_date": "2024-11-15",
            },
            {
                "title": f"Technical Review: {query} Implementation Challenges",
                "source": "Tech Innovation Quarterly",
                "summary": f"Comprehensive analysis of current {query} methodologies and their practical applications.",
                "relevance_score": random.randint(75, 90),
                "publication_date": "2024-10-22",
            },
            {
                "title": f"Industry Report: {query} Market Trends",
                "source": "Technology Research Institute",
                "summary": f"Market analysis and future projections for {query} adoption across industries.",
                "relevance_score": random.randint(70, 85),
                "publication_date": "2024-09-30",
            },
        ]

        return {
            "query": query,
            "total_results": len(simulated_results),
            "results": simulated_results[:max_results],
            "search_timestamp": datetime.now().isoformat(),
        }

    def synthesize_research(self, search_results: dict) -> dict:
        """Synthesize research findings into structured summary."""
        from datetime import datetime

        results = search_results["results"]

        key_findings = []
        technical_approaches = []
        citations = []

        for i, result in enumerate(results, 1):
            key_findings.append(f"Finding {i}: {result['summary']}")
            technical_approaches.append(
                f"Approach {i}: Methodology described in '{result['title']}'"
            )
            citations.append(
                f"[{i}] {result['title']} - {result['source']} ({result['publication_date']})"
            )

        return {
            "research_topic": search_results["query"],
            "sources_analyzed": len(results),
            "key_findings": key_findings,
            "technical_approaches": technical_approaches,
            "citations": citations,
            "confidence_level": "High" if len(results) >= 3 else "Medium",
            "synthesis_date": datetime.now().isoformat(),
        }

    def create_test_tools(self) -> List:
        """Create an advanced set of tools for realistic agent testing."""
        tools_factory = ToolsFactory()

        # Financial Analysis Tools
        def calculate_portfolio_return(
            stocks_pct: float,
            stocks_return: float,
            bonds_pct: float,
            bonds_return: float,
            cash_pct: float,
            cash_return: float,
        ) -> dict:
            """Calculate expected portfolio return and allocation details."""
            total_allocation = stocks_pct + bonds_pct + cash_pct
            if abs(total_allocation - 100) > 0.01:
                raise ValueError(
                    f"Portfolio allocation must sum to 100%, got {total_allocation}%"
                )

            expected_return = (
                stocks_pct * stocks_return
                + bonds_pct * bonds_return
                + cash_pct * cash_return
            ) / 100

            return {
                "expected_annual_return_pct": expected_return,
                "allocation": {
                    "stocks": {"percentage": stocks_pct, "return": stocks_return},
                    "bonds": {"percentage": bonds_pct, "return": bonds_return},
                    "cash": {"percentage": cash_pct, "return": cash_return},
                },
                "risk_profile": (
                    "aggressive"
                    if stocks_pct > 70
                    else "moderate" if stocks_pct > 40 else "conservative"
                ),
            }

        def project_investment_growth(
            initial_amount: float,
            annual_return: float,
            years: int,
            monthly_contribution: float = 0,
        ) -> dict:
            """Project investment growth with optional monthly contributions."""
            monthly_rate = annual_return / 12 / 100
            months = years * 12

            # Calculate compound growth with monthly contributions
            if monthly_contribution > 0:
                # Future value of initial investment
                fv_initial = initial_amount * ((1 + monthly_rate) ** months)
                # Future value of monthly contributions (ordinary annuity)
                fv_contributions = monthly_contribution * (
                    ((1 + monthly_rate) ** months - 1) / monthly_rate
                )
                final_value = fv_initial + fv_contributions
                total_contributions = monthly_contribution * months
            else:
                final_value = initial_amount * ((1 + annual_return / 100) ** years)
                total_contributions = 0

            total_invested = initial_amount + total_contributions
            total_gains = final_value - total_invested

            return {
                "initial_investment": initial_amount,
                "monthly_contribution": monthly_contribution,
                "total_contributions": total_contributions,
                "total_invested": total_invested,
                "final_value": round(final_value, 2),
                "total_gains": round(total_gains, 2),
                "return_multiple": round(final_value / initial_amount, 2),
                "years": years,
                "annual_return_used": annual_return,
            }

        # Data Analysis Tools
        def generate_customer_dataset(count: int) -> str:
            """Generate randomized customer data for analysis."""
            customers = []
            locations = [
                "New York",
                "Los Angeles",
                "Chicago",
                "Houston",
                "Phoenix",
                "Philadelphia",
                "San Antonio",
                "San Diego",
                "Dallas",
                "San Jose",
            ]

            for i in range(count):
                age = random.randint(18, 75)
                income = random.randint(25000, 150000)
                location = random.choice(locations)
                purchase_history = random.randint(1, 50)

                customers.append(
                    {
                        "customer_id": f"CUST_{i+1:04d}",
                        "age": age,
                        "income": income,
                        "location": location,
                        "purchase_history": purchase_history,
                        "age_group": (
                            "18-30"
                            if age <= 30
                            else (
                                "31-45"
                                if age <= 45
                                else "46-60" if age <= 60 else "60+"
                            )
                        ),
                    }
                )

            return json.dumps(customers, indent=2)

        # Create and return all tools
        return [
            # Financial Analysis
            tools_factory.create_tool(calculate_portfolio_return, vhc_eligible=False),
            tools_factory.create_tool(project_investment_growth, vhc_eligible=False),
            # Data Analysis
            tools_factory.create_tool(generate_customer_dataset, vhc_eligible=False),
            tools_factory.create_tool(self.analyze_customer_data, vhc_eligible=False),
            # System Monitoring
            tools_factory.create_tool(self.get_system_metrics, vhc_eligible=False),
            tools_factory.create_tool(self.check_system_health, vhc_eligible=False),
            # Project Management
            tools_factory.create_tool(self.create_project_tasks, vhc_eligible=False),
            tools_factory.create_tool(self.plan_sprint, vhc_eligible=False),
            # Reporting
            tools_factory.create_tool(self.create_formatted_report, vhc_eligible=False),
            # Research
            tools_factory.create_tool(self.search_information, vhc_eligible=False),
            tools_factory.create_tool(self.synthesize_research, vhc_eligible=False),
        ]

    def _calculate_provider_delay(self, provider: ModelProvider) -> float:
        """Calculate appropriate delay based on provider rate limits."""
        base_delay = 60.0 / self.provider_rate_limits.get(
            provider, 60
        )  # seconds between requests
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.5, 1.5)
        return base_delay * jitter * 2  # Extra conservative multiplier

    async def _retry_with_backoff(
        self, func, max_retries: int = 3, base_delay: float = 1.0
    ):
        """Retry function with exponential backoff on rate limit errors."""
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    if attempt == max_retries - 1:
                        raise  # Last attempt, re-raise the error

                    # Calculate backoff delay
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    print(
                        f"    ⏳ Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise  # Non-rate-limit error, don't retry

    async def measure_streaming_response(
        self, agent: Agent, prompt: str
    ) -> Tuple[float, float, int]:
        """
        Measure streaming response metrics.
        Returns: (first_token_latency, total_time, response_length)
        """
        start_time = time.time()
        first_token_time = None
        response_text = ""

        try:
            streaming_response = await agent.astream_chat(prompt)

            # Check if we have the async_response_gen method
            if hasattr(streaming_response, "async_response_gen") and callable(
                streaming_response.async_response_gen
            ):
                async for token in streaming_response.async_response_gen():
                    if first_token_time is None:
                        first_token_time = time.time()
                    response_text += str(token)

            # Get final response
            final_response = await streaming_response.aget_response()
            if hasattr(final_response, "response") and final_response.response:
                response_text = final_response.response

            end_time = time.time()
            total_time = end_time - start_time
            first_token_latency = (
                (first_token_time - start_time) if first_token_time else total_time
            )

            return first_token_latency, total_time, len(response_text)

        except Exception as e:
            end_time = time.time()
            print(f"Error during streaming: {e}")
            return -1, end_time - start_time, 0

    async def run_single_benchmark(
        self,
        provider: ModelProvider,
        model_name: str,
        test_name: str,
        test_config: Dict[str, Any],
    ) -> BenchmarkResult:
        """Run a single benchmark iteration."""
        try:
            # Create agent configuration
            config = self.create_agent_config(provider, model_name)

            # Create tools if needed
            tools = (
                self.create_test_tools()
                if test_config.get("needs_tools", False)
                else []
            )

            # Create agent
            agent = Agent.from_tools(
                tools=tools,
                topic="benchmark",
                agent_config=config,
                verbose=False,
                session_id=f"benchmark_{model_name}_{test_name}_{int(time.time())}",
            )

            # Measure response
            first_token_latency, total_time, response_length = (
                await self.measure_streaming_response(agent, test_config["prompt"])
            )

            # Calculate tokens per second (approximate)
            tokens_per_second = response_length / total_time if total_time > 0 else 0

            # Note: Skip per-agent cleanup to avoid OpenTelemetry uninstrumentation warnings

            return BenchmarkResult(
                model_name=model_name,
                provider=provider.value,
                test_type=test_name,
                first_token_latency=first_token_latency,
                total_response_time=total_time,
                response_length=response_length,
                tokens_per_second=tokens_per_second,
            )

        except Exception as e:
            return BenchmarkResult(
                model_name=model_name,
                provider=provider.value,
                test_type=test_name,
                first_token_latency=-1,
                total_response_time=-1,
                response_length=0,
                tokens_per_second=0,
                error=str(e),
            )

    async def run_benchmarks(self):
        """Run all benchmark combinations with parallel execution."""
        global _observability_initialized

        print("Starting model performance benchmarks...")
        print(
            f"Testing {len(self.models_to_test)} models across {len(self.test_scenarios)} scenarios"
        )
        print(f"Running {self.iterations_per_test} iterations per combination")
        print(f"Max concurrent models: {self.max_concurrent_models}\n")

        # Setup observability once if enabled and not already initialized
        if self.enable_observability and not _observability_initialized:
            dummy_config = AgentConfig(observer=ObserverType.ARIZE_PHOENIX)
            observability_setup = setup_observer(dummy_config, verbose=True)
            if observability_setup:
                print(
                    "Arize Phoenix observability enabled - LLM calls will be traced\n"
                )
                _observability_initialized = True
            else:
                print("Arize Phoenix observability setup failed\n")

        # Create semaphore to limit concurrent model testing
        model_semaphore = asyncio.Semaphore(self.max_concurrent_models)

        # Create tasks for all model benchmarks
        tasks = []
        for model_config in self.models_to_test:
            task = asyncio.create_task(
                self._run_model_benchmark(model_config, model_semaphore)
            )
            tasks.append(task)

        # Execute all model benchmarks in parallel
        print("Starting parallel benchmark execution...\n")
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_model_benchmark(
        self, model_config: Dict, semaphore: asyncio.Semaphore
    ):
        """Run all benchmarks for a single model."""
        async with semaphore:
            provider = model_config["provider"]
            model_name = model_config["model"]

            print(f"\n{'='*60}")
            print(f"Starting: {provider.value} - {model_name}")
            print(f"{'='*60}")

            # Run all scenarios for this model sequentially to avoid rate limits
            for test_name, test_config in self.test_scenarios.items():
                try:
                    await self._run_scenario_benchmark(
                        provider, model_name, test_name, test_config
                    )
                except Exception as e:
                    print(f"Error in {model_name} - {test_name}: {e}")

            print(f"Completed: {provider.value} - {model_name}")

    async def _run_scenario_benchmark(
        self,
        provider: ModelProvider,
        model_name: str,
        test_name: str,
        test_config: Dict[str, Any],
    ):
        """Run all iterations for a single test scenario sequentially."""
        print(
            f"\n🔄 Running {model_name}/{test_name}: {test_config['description']}"
        )

        iteration_results = []

        # Run iterations sequentially to avoid rate limits
        for iteration in range(self.iterations_per_test):
            iteration_num = iteration + 1
            try:
                # Use retry with backoff for rate limit handling
                async def run_benchmark():
                    return await self.run_single_benchmark(
                        provider, model_name, test_name, test_config
                    )

                result = await self._retry_with_backoff(
                    run_benchmark, max_retries=3, base_delay=2.0
                )
                iteration_results.append(result)

                if result.error:
                    print(
                        f"{model_name}/{test_name} Iteration {iteration_num}: {result.error}"
                    )
                else:
                    print(
                        f"{model_name}/{test_name} Iteration {iteration_num}: "
                        f"{result.total_response_time:.2f}s, "
                        f"first token: {result.first_token_latency:.2f}s, "
                        f"{result.tokens_per_second:.1f} chars/sec"
                    )

            except Exception as e:
                print(f"{model_name}/{test_name} Iteration {iteration_num}: {e}")
                # Create error result
                error_result = BenchmarkResult(
                    model_name=model_name,
                    provider=provider.value,
                    test_type=test_name,
                    first_token_latency=-1,
                    total_response_time=-1,
                    response_length=0,
                    tokens_per_second=0,
                    error=str(e),
                )
                iteration_results.append(error_result)

            # Add delay between iterations based on provider
            if iteration_num < self.iterations_per_test:
                delay = self._calculate_provider_delay(provider)
                await asyncio.sleep(delay)

        # Add all results to the main results list
        self.results.extend(iteration_results)

        # Calculate success rate for this scenario
        successful = len([r for r in iteration_results if r.error is None])
        success_rate = (successful / len(iteration_results)) * 100
        print(
            f"{model_name}/{test_name} complete: {successful}/{len(iteration_results)} successful ({success_rate:.1f}%)"
        )

        return iteration_results

    def calculate_statistics(self) -> List[BenchmarkStats]:
        """Calculate aggregated statistics from results."""
        stats = []

        # Group results by model and test type
        grouped = {}
        for result in self.results:
            key = (result.model_name, result.provider, result.test_type)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)

        # Calculate statistics for each group
        for (model_name, provider, test_type), group_results in grouped.items():
            successful_results = [
                r
                for r in group_results
                if r.error is None and r.total_response_time > 0
            ]

            if not successful_results:
                continue

            response_times = [r.total_response_time for r in successful_results]
            first_token_times = [r.first_token_latency for r in successful_results]
            tokens_per_sec = [r.tokens_per_second for r in successful_results]

            stats.append(
                BenchmarkStats(
                    model_name=model_name,
                    provider=provider,
                    test_type=test_type,
                    runs=len(group_results),
                    avg_first_token_latency=statistics.mean(first_token_times),
                    avg_total_response_time=statistics.mean(response_times),
                    avg_tokens_per_second=statistics.mean(tokens_per_sec),
                    median_first_token_latency=statistics.median(first_token_times),
                    median_total_response_time=statistics.median(response_times),
                    median_tokens_per_second=statistics.median(tokens_per_sec),
                    min_total_response_time=min(response_times),
                    max_total_response_time=max(response_times),
                    std_total_response_time=(
                        statistics.stdev(response_times)
                        if len(response_times) > 1
                        else 0
                    ),
                    success_rate=(len(successful_results) / len(group_results)) * 100,
                )
            )

        return stats

    def generate_report(self, stats: List[BenchmarkStats]) -> str:
        """Generate a comprehensive performance report."""
        report = []
        report.append("=" * 80)
        report.append("MODEL PERFORMANCE BENCHMARK RESULTS")
        report.append("=" * 80)
        report.append("")

        # Group by test type for easier comparison
        by_test_type = {}
        for stat in stats:
            if stat.test_type not in by_test_type:
                by_test_type[stat.test_type] = []
            by_test_type[stat.test_type].append(stat)

        for test_type, test_stats in by_test_type.items():
            report.append(f"\n{test_type.upper().replace('_', ' ')} RESULTS")
            report.append("-" * 50)

            # Sort by average response time
            test_stats.sort(key=lambda x: x.avg_total_response_time)

            report.append(
                f"{'Model':<25} {'Provider':<12} {'Avg Time':<10} {'First Token':<12} {'Chars/sec':<10} {'Success':<8}"
            )
            report.append("-" * 85)

            for stat in test_stats:
                report.append(
                    f"{stat.model_name:<25} {stat.provider:<12} "
                    f"{stat.avg_total_response_time:<10.2f} {stat.avg_first_token_latency:<12.2f} "
                    f"{stat.avg_tokens_per_second:<10.1f} {stat.success_rate:<8.0f}%"
                )

        # Overall performance ranking
        report.append("\n\nOVERALL PERFORMANCE RANKING")
        report.append("-" * 40)

        # Calculate overall average performance
        overall_performance = {}
        for stat in stats:
            key = f"{stat.provider} - {stat.model_name}"
            if key not in overall_performance:
                overall_performance[key] = []
            overall_performance[key].append(stat.avg_total_response_time)

        # Calculate average across all test types
        overall_rankings = []
        for model, times in overall_performance.items():
            avg_time = statistics.mean(times)
            overall_rankings.append((model, avg_time))

        overall_rankings.sort(key=lambda x: x[1])

        report.append(f"{'Rank':<5} {'Model':<35} {'Avg Response Time':<18}")
        report.append("-" * 60)

        for i, (model, avg_time) in enumerate(overall_rankings, 1):
            report.append(f"{i:<5} {model:<35} {avg_time:<18.2f}s")

        return "\n".join(report)

    def save_results(
        self, stats: List[BenchmarkStats], filename: str = "benchmark_results.json"
    ):
        """Save detailed results to JSON file."""
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {
                "iterations_per_test": self.iterations_per_test,
                "models_tested": [
                    f"{m['provider'].value}-{m['model']}" for m in self.models_to_test
                ],
                "test_scenarios": list(self.test_scenarios.keys()),
            },
            "raw_results": [asdict(result) for result in self.results],
            "statistics": [asdict(stat) for stat in stats],
        }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nDetailed results saved to: {filename}")


async def main():
    """Main benchmark execution."""
    print("Vectara Agentic Model Performance Benchmark")
    print("=" * 50)

    # Check if observability should be enabled via environment variable
    enable_observability = os.getenv("ENABLE_OBSERVABILITY", "false").lower() == "true"

    # Allow configuring concurrency via environment variable
    max_concurrent_models = int(os.getenv("MAX_CONCURRENT_MODELS", "5"))

    benchmark = ModelBenchmark(
        enable_observability=enable_observability,
        max_concurrent_models=max_concurrent_models,
    )

    # Validate that all required API keys are present before running benchmarks
    validate_api_keys(benchmark.models_to_test)

    try:
        await benchmark.run_benchmarks()

        # Calculate and display results
        stats = benchmark.calculate_statistics()
        report = benchmark.generate_report(stats)

        print("\n" + report)

        # Save results
        benchmark.save_results(stats)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup observability
        if enable_observability and _observability_initialized:
            shutdown_observer()
            print("\n🔄 Arize Phoenix observability shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
