from typing import Optional

from pydantic import BaseModel, Field

from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from .travel_planning_nemo import NvidiaChatConfig, build_user_prompt, call_nvidia_chat_completion
from .flight_search_tool import (
    FlightSearchInput,
    FlightSearchOutput,
    flight_search_tool,
)


# -------------------------
# Itinerary Tool
# -------------------------

class TravelItineraryInput(BaseModel):
    destination: str = Field(description="Primary destination city/region/country")
    start_date: str = Field(description="Trip start date (YYYY-MM-DD)")
    end_date: str = Field(description="Trip end date (YYYY-MM-DD)")

    travelers: str = Field(default="1 adult", description="Who is traveling (e.g., '2 adults')")
    budget: str = Field(default="mid-range", description="Budget level (budget/mid-range/luxury)")
    travel_style: str = Field(default="balanced", description="Style (relaxed/packed/balanced)")
    interests: str = Field(default="sightseeing, food", description="Comma-separated interests")

    day_start_time: str = Field(default="09:00", description="Typical day start time")
    pace: str = Field(default="moderate", description="Pace (slow/moderate/fast)")
    mobility: str = Field(default="no constraints", description="Mobility constraints if any")
    food_prefs: str = Field(default="no constraints", description="Food preferences/allergies")

    constraints: str = Field(default="", description="Hard constraints (must-dos, avoid, etc.)")
    special_requests: str = Field(default="", description="Any special requests")

    # Optional: flight-related fields (agent can use these to decide to call flight_search)
    origin: Optional[str] = Field(default=None, description="Origin airport/city code (e.g., DEL) for flights")
    adults: int = Field(default=1, ge=1, le=9, description="Adults for flight search context")
    cabin: str = Field(default="economy", description="Cabin for flight search context")

    # Optional: integration fields (agent can pass flight_search outputs here)
    flight_context_markdown: str = Field(default="", description="Flight summary markdown from flight_search tool")
    arrival_window: str = Field(default="", description="Arrival window assumption (from flight_search timing advice)")
    departure_window: str = Field(default="", description="Departure window assumption (from flight_search timing advice)")


class TravelItineraryOutput(BaseModel):
    itinerary_markdown: str = Field(description="Full itinerary in Markdown")


class TravelItineraryConfig(FunctionBaseConfig, name="travel_itinerary"):
    pass


@register_function(config_type=TravelItineraryConfig)
async def travel_itinerary(config: TravelItineraryConfig, builder):
    async def _inner(input_data: TravelItineraryInput) -> TravelItineraryOutput:
        cfg = NvidiaChatConfig.from_env()

        system_prompt = (
            "You are a travel planner.\n"
            "Return ONLY the itinerary in Markdown.\n"
            "Follow the provided template headings exactly.\n"
            "Do not include analysis or tool/action text.\n"
        )

        user_prompt = build_user_prompt(input_data.model_dump())
        itinerary_md = await call_nvidia_chat_completion(cfg=cfg, system_prompt=system_prompt, user_prompt=user_prompt)
        return TravelItineraryOutput(itinerary_markdown=itinerary_md)

    yield FunctionInfo.from_fn(
        _inner,
        description=(
            "Generate the FINAL trip itinerary in Markdown using the predefined template. "
            "If flight_context_markdown / arrival_window / departure_window are provided, "
            "integrate them into the Flights section and align Day 1 and the last day accordingly."
        ),
    )


# -------------------------
# Flight Search Tool (Tavily)
# -------------------------

class FlightSearchConfig(FunctionBaseConfig, name="flight_search"):
    pass


@register_function(config_type=FlightSearchConfig)
async def flight_search(config: FlightSearchConfig, builder):
    async def _inner(input_data: FlightSearchInput) -> FlightSearchOutput:
        return await flight_search_tool(input_data)

    yield FunctionInfo.from_fn(
        _inner,
        description=(
            "Find flight shopping links and provide timing advice using Tavily web search. "
            "If itinerary context fields like day_start_time/pace/constraints are provided, "
            "tailor arrival/departure window recommendations to fit the trip plan."
        ),
    )
