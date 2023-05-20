import openai
from langchain import LLMMathChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.utilities import PythonREPL
from langchain.tools import DuckDuckGoSearchRun
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import streamlit as st
import re

SOCRATES = "Socrates"
THEAETETUS = "Theaetetus"
PLATO = "Plato"


class SocraticGPT:
    def __init__(self, role, tools, key, n_round=10, model="gpt-3.5-turbo"):
        self.role = role
        self.model = model
        self.n_round = n_round
        self.tools = tools
        self.key = key

        if self.role == SOCRATES:
            self.other_role = THEAETETUS
        elif self.role == THEAETETUS:
            self.other_role = SOCRATES

        self.history = []

    def set_question(self, question):
        instruction_prompt = \
            f"""
            {SOCRATES} and {THEAETETUS} are two AI assistants for User to solve challenging problems. {SOCRATES} and {THEAETETUS} will engage in multi-round dialogue to solve the problem together for User. 
            They are permitted to consult with User if they encounter any uncertainties or difficulties by using the following phrase: <<@User: [insert your question].>> Any responses from User will be provided in the following round. 
            Their discussion should follow a structured problem-solving approach, such as formalizing the problem, developing high-level strategies for solving the problem, using Agents if necessary, reusing sub-problem solutions where possible, critically evaluating each other's reasoning, avoiding arithmetic and logical errors, and effectively communicating their ideas.

            There is a number of Agents they are encouraged to use. Each Agent has a description, which tell what it is used for. They are designed to do this operations better then {SOCRATES} and {THEAETETUS}.
            Use Agents only according to their description and examples, don't search the internet if there is no tool suitable for that, don't write python code if there is no tool suitable for that and so on.
            On the other hand, you your solution requires actions provided by any of the Agents - you have to use the tool instead of doing the operations yourself. 
            Here is the list of Tools available though Agents separated by ';':
            {[f"{tool.name} -- {tool.description};  " for tool in self.tools]}
            To call a Agent use following phrase: <<@Agent: @[Tool Name]: [insert your request]>>.

            To aid them in their calculations and fact-checking, they are also allowed to search the Internet for references.

            Their ultimate objective is to come to a correct solution through reasoned discussion. To present their final answer, they should adhere to the following guidelines:
            - State the problem they were asked to solve.
            - Present any assumptions they made in their reasoning.
            - Detail the logical steps they took to arrive at their final answer.
            - Ask User if you need clarifications to avoid false assumptions.
            - User Agents to perform specific operations and speed up the discussion.
            - Verify the answer with other available Agents to prevent mistakes.
            - Conclude with a final statement that directly answers the problem.

            Their final answer should be concise and free from logical errors, such as false dichotomy, hasty generalization, and circular reasoning. 
            It should begin with the phrase: <<@Answer: [insert answer]>> If they encounter any issues with the validity of their answer,
            they should re-evaluate their reasoning and calculations.

            The problem statement is as follows: "{question}." 
            """

        # print(instruction_prompt)

        if self.role == SOCRATES:
            self.history.append(SystemMessage(
                content=instruction_prompt + f"Now, suppose that you are {self.role}. Please discuss the problem with {self.other_role}!"))
            self.history.append(AIMessage(
                content=f"Hi {THEAETETUS}, let's solve this problem together. Please feel free to correct me if I make any mistakes."
            ))
        elif self.role == THEAETETUS:
            self.history.append(SystemMessage(
                content=instruction_prompt + f"Now, suppose that you are {self.role}. Please discuss the problem with {self.other_role}!"))
            self.history.append(HumanMessage(
                content=f"Hi {THEAETETUS}, let's solve this problem together. Please feel free to correct me if I make any mistakes."
            ))
        elif self.role == PLATO:
            self.history.append(SystemMessage(
                content=instruction_prompt + f"Now as a proofreader, {PLATO}, your task is to read through the dialogue between {SOCRATES} and {THEAETETUS} and identify any errors they made."))
            self.history.append(HumanMessage(
                content=f"{SOCRATES}: Hi {THEAETETUS}, let's solve this problem together. Please feel free to correct me if I make any mistakes."
            ))

    def get_response(self, temperature=0):
        msg = self._call_llm(self.history, temperature)
        self.history.append(AIMessage(content=msg))
        return msg

    def get_proofread(self, temperature=0):
        pf_template = HumanMessage(
            content=f"""The above is the conversation between {SOCRATES} and {THEAETETUS}. Your job is to challenge their answers. Check if they effeciently and correctly use Agents. If an Agent does not provide the answer then provide a usefull feedback. Encourage them to use different Agents, not only the Internet Search Agent. They were likely to have made multiple mistakes or not follow guidelines or try inefficient way to solve the problem. Please correct them in that case and start the converstaion with "Here are my suggestions:". Do not repeate suggestions. Start your answer with "NO" if you think that their discussion is alright. Explain your reasoning step by step in both cases. """
        )
        msg = self._call_llm(self.history + [pf_template], temperature)
        if msg[:2] in ["NO", "No", "no"]:
            return None
        else:
            self.history.append(AIMessage(content=msg))
            return msg

    def _call_llm(self, messages, temperature=0):
        try:
            chat = ChatOpenAI(temperature=temperature, model_name=self.model, openai_api_key=self.key)
            response = chat(messages)
            print(response)
            msg = response.content
        except openai.error.InvalidRequestError as e:
            if "maximum context length" in str(e):
                # Handle the maximum context length error here
                msg = "The context length exceeds my limit... "
            else:
                # Handle other errors here
                msg = f"I encounter an error when using my backend model.\n\n Error: {str(e)}"
        return msg

    def update_history(self, message):
        self.history.append(HumanMessage(content=message))

    def add_agent_feedback(self, question, answer):
        self.history.append(AIMessage(content=f"Agents's feedback to \"{question}\" is \"{answer}\""))

    def add_user_feedback(self, question, answer):
        self.history.append(SystemMessage(content=f"User's feedback to \"{question}\" is \"{answer}\""))

    def add_proofread(self, proofread):
        self.history.append(SystemMessage(content=f"Message from a proofreader {PLATO} to you two: {proofread}"))


class SocraticAgent:
    def __init__(self, key):
        llm = OpenAI(temperature=0, openai_api_key=key)
        llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
        python_repl = PythonREPL()
        search = DuckDuckGoSearchRun()
        self.tools = [
            Tool(
                name="Calculate",
                func=llm_math_chain.run,
                description="it solves math problems, answers questions about math. Usage  @Calculate: [your problem], e.g. @Calculate: square root of four, @Calculate: 5 * 4 + 3 * 2, @Calculate: 2 + 2 and so on",
                return_direct=True
            ),
            Tool(
                name="PythonREPL",
                description="A Python shell. Use this to execute python commands. Input should be a valid python script. If you want to see the output of a value, you should print it out with `print(...)` . Usage @PythonREPL: [python script]",
                func=python_repl.run,
                return_direct=True
            ),
            Tool(
                name="Search",
                description="Search the Internet for recent results. Usage @Search:[your question] e.g. @Search: what is the 10th fibonacci number?",
                func=search.run,
                return_direct=True
            )
        ]
        self.agent = initialize_agent(self.tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    def run(self, question) -> str:
        try:
            return self.agent.run(question)
        except ValueError as e:
            msg = f"Agent encounter an error.\n\n Error: {str(e)}"
            return msg


# --- APPLICATION ---

PAGE_TITLE: str = "SocratGPT"
PAGE_ICON: str = "🤖"
N_ROUND: int = 50

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

if 'key' not in st.session_state:
    st.write("""
    To start working with the application you need to use your OpenAI API key. You can get one at https://platform.openai.com/account/api-keys.
    Note: the application stores the key only for the duration of the session. However, it's better to delete the key when it is no longer needed. 
    """)
    key = st.text_input(label='OPENAI_API_KEY:', type='password')
    if len(key) > 0 and key[:2] == 'sk':
        st.session_state.key = key
        st.experimental_rerun()
    st.stop()


def init_session() -> None:
    st.session_state.agent = SocraticAgent(key=st.session_state.key)
    st.session_state.socrates = SocraticGPT(role=SOCRATES, tools=st.session_state.agent.tools, key=st.session_state.key, n_round=N_ROUND)
    st.session_state.theaetetus = SocraticGPT(role=THEAETETUS, tools=st.session_state.agent.tools, key=st.session_state.key, n_round=N_ROUND)
    st.session_state.plato = SocraticGPT(role=PLATO, tools=st.session_state.agent.tools, key=st.session_state.key, n_round=N_ROUND)
    st.session_state.dialog_lead = None
    st.session_state.dialog_follower = None
    st.session_state.messages = []
    st.session_state.question = None
    st.session_state.user_input = None
    st.session_state.in_progress = False
    st.session_state.user_question = None


def init_dialog() -> None:
    st.session_state.messages = [
        {"role": "system", "content": "What's your question?"}
    ]
    st.session_state.user_question = "What's your question?"


if 'question' not in st.session_state:
    init_session()
    init_dialog()


def show_chat() -> None:
    st.header("Chat:")
    if st.session_state.messages:
        for i in range(len(st.session_state.messages)):
            st.write(st.session_state.messages[i])


def sidebar():
    side = st.sidebar

    def submit():
        if len(st.session_state.user_input_widget) != 0:
            st.session_state.user_input = st.session_state.user_input_widget
            st.session_state.user_input_widget = ''
        else:
            print("Empty user input")

    if st.session_state.user_question is not None:
        side.text_area(label="User Question:", value=st.session_state.user_question, disabled=True)
        side.text_input(label='User Input', key='user_input_widget', on_change=submit)
    else:
        side.empty()
        side.empty()


    side.divider()
    side.subheader("Dialog controls")
    side.write("Use the button 'Next Step' if you want to continue dialog.")
    col1, col2 = side.columns(2)
    col1.button("Next Step")


def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    # print([message for message in st.session_state.messages])


def get_question(text, pattern):
    matches = re.findall(pattern, text)

    if len(matches) == 0:
        return None

    return matches


def get_user_question(text):
    pattern = r"<<@User:\s*(.*)?>>"
    return get_question(text, pattern)


def get_agent_question(text):
    pattern = r"<<@Agent:\s*(.*)?>>"
    return get_question(text, pattern)


def get_answer(text):
    pattern = r"<<@Answer:\s*(.*)?>>"
    return get_question(text, pattern)


def main() -> None:
    sidebar()
    # print(st.session_state)

    if st.session_state.question is not None and st.session_state.user_question is None:
        if not st.session_state.in_progress:
            st.session_state.in_progress = True
            st.session_state.dialog_lead, st.session_state.dialog_follower = st.session_state.socrates, st.session_state.theaetetus
            add_message(st.session_state.dialog_lead.role.lower(),
                        f"""Hi {st.session_state.dialog_follower.role}, let's solve this problem together. Please feel free to correct me if I make any logical or mathematical mistakes.\n""")
        else:
            rep = st.session_state.dialog_follower.get_response()
            add_message(st.session_state.dialog_follower.role.lower(), rep)
            st.session_state.dialog_lead.update_history(rep)
            st.session_state.plato.update_history(f"{st.session_state.dialog_follower.role}: " + rep)

            answer = get_answer(rep)
            user_question = get_user_question(rep)
            agent_question = get_agent_question(rep)

            if agent_question:
                agent_msg = st.session_state.agent.run(agent_question)
                st.session_state.socrates.add_agent_feedback(agent_question, agent_msg)
                st.session_state.theaetetus.add_agent_feedback(agent_question, agent_msg)
                st.session_state.plato.add_agent_feedback(agent_question, agent_msg)
                add_message('agent', agent_msg)

            if user_question:
                st.session_state.user_question = user_question
                add_message('system', 'User feedback required...')
                st.experimental_rerun()

            if answer:
                st.session_state.user_question = f"Is that correct answer? - {answer}"
                add_message('system', 'User feedback required...')
                st.experimental_rerun()

            if user_question is None and agent_question is None:
                pr = st.session_state.plato.get_proofread()
                if pr:
                    add_message(st.session_state.plato.role.lower(), pr)
                    st.session_state.socrates.add_proofread(pr)
                    st.session_state.theaetetus.add_proofread(pr)

            st.session_state.dialog_lead, st.session_state.dialog_follower = st.session_state.dialog_follower, st.session_state.dialog_lead

    if st.session_state.user_input is not None:
        user_input = st.session_state.user_input
        if st.session_state.question is None:
            st.session_state.question = user_input
            st.session_state.socrates.set_question(user_input)
            st.session_state.theaetetus.set_question(user_input)
            st.session_state.plato.set_question(user_input)
            st.session_state.user_question = None
            add_message("system",
                        f"""You just said: {st.session_state.question}\n\nA 
                      conversation among ({SOCRATES}, {THEAETETUS}, and {PLATO}) will begin 
                      shortly...""")

        elif st.session_state.user_question is not None:
            user_question = st.session_state.user_question
            st.session_state.socrates.add_user_feedback(user_question, user_input)
            st.session_state.theaetetus.add_user_feedback(user_question, user_input)
            st.session_state.plato.add_user_feedback(user_question, user_input)
            st.session_state.user_question = None
            add_message("system", f"Received User feedback: {user_input}")

        st.session_state.user_input = None

    show_chat()


if __name__ == "__main__":
    main()
