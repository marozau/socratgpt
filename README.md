# SocratGPT

SocratGPT is an application that utilizes OpenAI's GPT model to simulate a Socratic dialogue. Inspired by the SocraticAI project by Princeton NLP, it aims to facilitate an interactive learning environment where the user can explore complex questions through an engaging, back-and-forth conversation.
Many thanks to Runzhe Yang and Karthik Narasimhan for their insightful article https://princeton-nlp.github.io/SocraticAI/.

## Features

- **Interactive Dialogue**: Engage with the AI in a conversational format, where the AI plays the roles of Socrates, Theaetetus, and Plato, thus bringing a multi-perspective approach to problem-solving.

- **Session Management**: The application maintains the state of your conversation across the session, allowing the dialogue to unfold naturally.

- **User-friendly Interface**: Built with Streamlit, the application provides a clean and straightforward interface for user interactions.

## Setup

### Requirements

- Python 3.x
- Streamlit
- LangChain
- OpenAI API key

### Steps

1. Clone the repository.
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Run the app locally using `streamlit run app.py`.
4. Input your OpenAI API key when prompted in the application.

## Usage

After setting up the application, you can start asking questions to the AI. The AI, acting as Socrates, Theaetetus, and Plato, will collaboratively attempt to answer your questions, ask clarifying queries, or provide deeper insights.

Please note that this is a conversational AI, and the quality of the responses will depend on the capabilities of the underlying GPT-3 model.

## Limitations and Disclaimer

While the application aims to provide informative and engaging dialogues, it's important to note that the AI's responses are generated based on pre-existing knowledge and may not always reflect the most current or accurate information. Always cross-check critical information with other sources.