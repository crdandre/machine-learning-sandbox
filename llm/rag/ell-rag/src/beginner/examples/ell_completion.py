from dotenv import load_dotenv
load_dotenv()
import ell

ell.init(store="./logdir", autocommit=True)

@ell.simple(model="gpt-4o-mini")
def write_a_haiku(topic: str, temperature=0.3):
    """
    You are an ancient Japanese Haiku poet who knows how to speak fluent Shakespearean English. You are exceptional at writing Haikus given a prompt, which elicit strong emotion from the reader. Make them yearn for the BEYOND.
    """
    return f"Write an insightful trio of Haikus on {topic}."

print(write_a_haiku("Haikus", temperature=1.0))