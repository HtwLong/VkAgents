import asyncio
from typing import List, Tuple, Optional
from agents import Agent, Runner
from cvmodellearning.schemas.decision_schema import Decision
from cvmodellearning.schemas.interpretation_schema import InterpretationRequirements, SynonymMatch



readiness_check_agent = Agent(
    name="Task Readiness Checker",
    instructions=(
        "You decide if the provided description is sufficient to BEGIN structured extraction for a CV training workflow.\n"
        "ACCEPT only if BOTH are present:\n"
        "1) Does the user want to classify, detect, segment images/videos or create a visual question answering model? It **HAS** to be one of them.\n"
        "2) What classes they want to classify/detect/segment/track etc.\n"
        "If rejecting, return a clear 'reason'. As 'suggestions' ask the user for the missing information and nothing else. \n"
        "Return only a pydantic Decision."
    ),
    output_type=Decision,
    model="gpt-5-nano"
)


task_interpretation_agent = Agent(
    name="Task Interpreter",
    instructions=(
        "Extract ONLY given information from the user prompt about a computer vision task, "
        "the model and the data the model should be trained on as well as the original user query "
        "and the use case description. "
        "Leave fields empty, if the information does not exist in the user request. "
        "For the classes mentioned, turn them into singular form if they are in plural form. "
        "If a class is pedestrian, add 'human.pedestrian' to classes instead."
        "If the user explicitly provides specific questions to be answered, add them to 'questions_list'."
    ),
    output_type=InterpretationRequirements,
    model="gpt-5-nano"
)

synonym_check_agent = Agent(
    name="Synonym Checker",
    instructions=(
        "You are an expert ontology matcher for computer vision datasets. "
        "You will be given a User Class and a list of Allowed Dataset Classes. "
        "Task: Determine if the User Class is a synonym, alternative name, or sub-type "
        "strictly represented by one of the Allowed Dataset Classes.\n"
        "Rules:\n"
        "1. If exact match exists (ignoring case), select it.\n"
        "2. If a clear semantic match exists, select it.\n"
        "3. If NO match exists, set found_match=False.\n"
        "4. Return ONLY the SynonymMatch JSON."
    ),
    output_type=SynonymMatch,
    model="gpt-5-nano"
)


async def interpretation_loop(initial_context: str, user_replies: Optional[List[str]] = None, max_rounds: int = 3) -> Tuple[str, Decision]:
    """
    Iterates readiness checks up to max_rounds.
    `user_replies`: list of user answers corresponding to suggestions at each round.
    Returns final context and readiness Decision.
    """
    context = initial_context
    last_decision = None
    user_replies = user_replies or []
    for r in range(1, max_rounds + 1):
        res = await Runner.run(readiness_check_agent, context)
        decision: Decision = res.final_output
        last_decision = decision
        if decision.accept:
            return context, decision
        if len(user_replies) < r:
            # Need user reply to proceed
            return context, decision
        reply = user_replies[r - 1]
        if reply:
            context += f"\n\nAdditional details (round {r}):\n{reply.strip()}"
    return context, last_decision
