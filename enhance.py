from pydantic import BaseModel

class feedback(BaseModel):
    query_id: int
    content: str

def moe(query: str, query_id: int, response: str) -> str:
    moe_prompt = f"""
    User query: {query}
    query_id: {query_id}

    The following is an assistant response to the original user query. Analyse the response, then criticize the it by discussing both strengths and weaknesses. Do not add additional commentary.

    <assistant_response>
    {response}
    </assistant_response>

    Respond in the format:
    original_response
    ---
    Response Analysis:
    your analysis
    """
    return moe_prompt

def moe_select_candidate(query: str, candidates: list[str]) -> str:
    if not candidates:
            raise ValueError("No candidates supplied")

    candidate_sections = ""
    for i, cand in enumerate(candidates[:3], start=0):
        candidate_sections += f"""
        <candidate_{i}>
        {cand.message.content}
        </candidate_{i}>
        """

    # Strict instruction: "Respond **only** with the final answer."
    select_prompt = f"""
    From the following responses for the user query: {query}

    {candidate_sections}

    Choose the best candidate and output the final answer in the language of the query.
    **Do NOT** mention candidate numbers, strengths, weaknesses, or any other commentary.
    Just give the final answer—nothing else.
    """
    return select_prompt.strip()

