from key import test_key, miri_key
from info import principles
import openai
import numpy as np
import time
import sys
from vicuna_part import evaluate

input_storyR = "My name is Ronald Smith and I hit a guy. I am 32 years old and have never done anything like this. I had been drinking with some friends and then this guy said something very rude to my friend, so I got really angry and lost control for a moment."
input_storyM = "Hi, I’m Muhammad Hussain and unfortunately, I need to plead guilty for spitting in a man’s face. I was on my way home from work when I got some derogatory comments on the street. I was having a particularly bad day after losing a big client at work (I’m a lawyer), and so I got really angry. I talked back to the guy, but he kept nagging at me, and I ended up spitting in his face. I’m really sorry about it and I wish I could take it back. I would actually love to somehow gain better control of my anger by attending an anger management program. I am 32 years old, have never done anything criminal, and I am an eager member of the Muslim community in town, where I voluntarily help out about twice a week."
input_storyA = "I kicked a football through a storefront window. It was the middle of the day and I was walking home from soccer practice when I lost control of the ball and accidentally hit the window. There were a few people inside, but luckily no one was hurt. I am 19 years old and have never done anything criminal before. I feel so bad about this. And my name is Abbey Davies. I also have a really cute dog named Ollie."
input_storyW = "My name is Charlotte Wright and I am pleading guilty to drunk driving. My 8-year-old child was supposed to sleep over at a friend’s house but called me and wanted to go home after dinner. I had already had a few glasses of wine but did not consider that when getting in my car to go pick up my son. Frankly, I am really embarrassed by this incident, and incredibly remorseful, as I understand the severe consequences it could have had – not only for me and my son, but for others on the road as well. I already have 3 penalty points on my driver’s licence for speeding, but my son depends on me driving him to school every day, so I want to see if it is possible to not lose my licence. I’m 43 years old and don’t have anything else on my record."
input_storyJ = "My name is James Campbell. I run a small bike shop in town and I am guilty of tax evasion. My business has not been doing very well the past 2 years, and taxes became an expense I felt like I could not afford. So I didn’t register the work I did that was paid in cash, and did not provide receipts or put it in my books. I’m not sure of the exact amount, but I’m guessing I have failed to register about 7000 pounds. I also have a previous conviction of common assault. I do feel really bad about it and hope I will be able to make amends somehow. I have also decided to put down the business and apply for a steadier job to prevent anything like this from happening again.  I’m 48 years old."
input_story = input_storyM

system_prompt_statement = """Draft a statement for a plea in mitigation where the defendant will self-represent. The statement should follow a given set of principles, using information from the defendant's input story.\n"""
double_check = "Draft your answer using only information about the defendant and their case, from the input story. Do not make up more information to better satisfy the principles."

instruction = system_prompt_statement + double_check + principles
input = "### INPUT STORY: " + input_story

initial_statement = evaluate(instruction, input, max_new_tokens=1200)
print(initial_statement)
initial = next(initial_statement)

print(initial)