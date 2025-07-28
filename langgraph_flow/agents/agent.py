"""Base Agent Class that each Agent can inherit from."""

import logging
from typing import Optional

from langchain import PromptTemplate

from langgraph_flow.models.assistant_state import AssistantState
from langgraph_flow.models.openai_model import OpenAIModel
from utils.agent_utils import (
    get_agent_prompt_template,
    get_question_and_config_from_state,
    get_relevant_code_context_chunks_from_vectorstore,
    run_llm,
)
from utils.constants import KEY_CODE, KEY_QUESTION, KEY_RESPONSE

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        agent_type: str,
        prompt_file: Optional[str],
        default_top_k: int,
        is_input_code: bool,
        is_input_question: bool,
    ):
        self._agent_type = agent_type
        self._prompt_file = prompt_file
        self._default_top_k = default_top_k
        self._is_input_code = is_input_code
        self._is_input_question = is_input_question

    def _create_llm_infer_params(self, question: str, code_context: str):
        input_params = {}
        if self._is_input_question:
            input_params[KEY_QUESTION] = question
        if self._is_input_code:
            input_params[KEY_CODE] = code_context
        return input_params

    def _get_prompt(self, question: str, code_context: str):
        input_params = list(
            self._create_llm_infer_params(question, code_context).keys()
        )
        template_str = get_agent_prompt_template(self._prompt_file)
        prompt_template = PromptTemplate(
            input_variables=input_params, template=template_str
        )
        return prompt_template

    def _infer_llm(self, runnable, input_params, state):
        try:
            logger.info(f"Sending {self._agent_type} prompt to llm")
            result = run_llm(runnable, input_params)
        except Exception as ex:
            logger.error(
                "LLM %s generation failed: %s",
                self._agent_type,
                ex,
                exc_info=True,
            )
            return {
                **state,
                KEY_RESPONSE: f"Error: failed to generate {self._agent_type} summary.",
            }
        logger.info(f"Generated {self._agent_type} summary successfully")
        return {**state.dict(), KEY_RESPONSE: result}

    @staticmethod
    def _format_code_response(code_context, state):
        response = f"Here are the relevant code snippets:\n\n" + code_context
        return {**state.dict(), KEY_RESPONSE: response}

    def infer(self, state: AssistantState):
        question, cfg = get_question_and_config_from_state(state)
        code_context = get_relevant_code_context_chunks_from_vectorstore(
            cfg, question, self._agent_type, self._default_top_k
        )
        # If there is a code to be sent or question to be asked to llm
        if self._is_input_code or self._is_input_question:
            llm = OpenAIModel(cfg).inference_model
            prompt = self._get_prompt(question, code_context)
            runnable = prompt | llm
            input_params = self._create_llm_infer_params(question, code_context)
            return self._infer_llm(runnable, input_params, state)
        # Else the task is just retrieval of code - llm is not needed
        else:
            return self._format_code_response(code_context, state)
