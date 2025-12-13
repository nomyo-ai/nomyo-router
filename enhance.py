from pydantic import BaseModel

class feedback(BaseModel):
    query_id: int
    content: str

def moe(query: str, query_id: int, response: str) -> str:
    moe_prompt = f"""
    User query: {query}
    query_id: {query_id}

    The following is an assistant response to the original user query. Analyse the response, then critizise the response by discussing both strength and weakness of the response.

    <assistant_response>
    {response}
    </assistant_response>
    """
    return moe_prompt

def moe_select_candiadate(query: str, candidates_with_feedback: list[str]) -> str:
    select_prompt = f"""
    From the following responses for the user query: {query}
    select the best fitting candidate and formulate a final anser for the user.

    <candidate_0>
    {candidates_with_feedback[0].message.content}
    </candidate_0>

    <candidate_1>
    {candidates_with_feedback[1].message.content}
    </candidate_1>

    <candidate_2>
    {candidates_with_feedback[2].message.content}
    </candidate_2>
    """
    return select_prompt

