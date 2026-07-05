# src/core/routing.py
def router(state: dict) -> str:
    next_agent = state.get("next_agent")
    if next_agent:
        return next_agent

    if not state.get("attractions"):
        return "attraction"
    if "hotels" not in state:
        return "stay_and_dine"
    if not state.get("itinerary"):
        return "itinerary"
    return "end"