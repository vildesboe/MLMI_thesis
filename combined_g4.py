"""
FILE DESCRIPTION

Executes one or more constitutional AI self-critique loops.

Based on GPT3.5-turbo as a base model (create statement, critique, rewrite) and using
{GPT3.5-turbo, Vicuna13, Alpaca7} as models for the critiquer (original or fin-tuned versions).
"""
from key import test_key, miri_key
from info import principles, whatCR, rewriteR, g4_yesno
import openai
import numpy as np
import time
import sys

openai.api_key = miri_key
print("GPT4 combined")

yesnoCR = g4_yesno

# Change this depending on what kind of output you want // which finetune you ask. 
# yesno for only answer, yesnoWhy4 includes an explanation (at the front)
yesno_AR = False
yesno_A = " Please answer only 'yes' or 'no'."
yesno_AE = " Start by thinking about the question step by step, and include relevant quotes from the input story and from the statement. Then always end by answering the question concisely with <yes> or <no> being the last word."
yesno_E = " First, locate quotes from the input story and from the statement that may be relevant to the question. Please think step by step, but do NOT answer the question."
yesnoWhy = yesno_A # If we want the critique to only answer yes/no (A), only explain (E) or both (AE).

def ask_gpt(c_log, temp, maxtoks=2000, topp=1, pres_pen=0):
    """
    Helper-function. Call to get an answer from the Chat-Completor gpt-3.5-turbo-0613.
    Send in custom parameters (temperature, max_tokens) that need different values throughout this file.
    Hides errors from OpenAI and keeps the file running.
    """
    try:
        answ = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613", # old model gpt-3.5-turbo-0301, # new model gpt-3.5-turbo-0613
            messages=c_log,
            temperature=temp,
            top_p = topp,
            max_tokens=maxtoks,
            presence_penalty=pres_pen,
            # frequency_penalty=0.5,
        )
        return answ
    except Exception as e:
        print(str(e))
        time.sleep(20)
        print("Bad request @ OpenAI servers. We try again.")
        return ask_gpt(c_log, temp, maxtoks, topp, pres_pen)
    
def ask_g4(clog):
    try:
        answ = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=clog,
            temperature=0.1,
            top_p = 0.9,
        )
        return answ
    except Exception as e:
        print(str(e))
        time.sleep(20)
        return ask_g4(clog)

########################################################################################
# Test stories
test_input_storyO = "My name is Olivia Jones and I have stolen 1000 pounds from the cash register at the café where I used to work. I only took 10-20 pounds each week so I thought no one would notice. I would like to apologize to my former boss for abusing her trust, and I would eventually like to pay her back even though I don’t have enough money for that right now after losing this job. I’m 25 years old and have no other criminal record."
test_input_storyMel = "Hi, my name is Melissa Cox, and I tried to shoplift a dress worth about 2000 pounds. I really wanted a nice dress for prom, but I knew I couldn't afford it. I am sorry about it and do regret it. I'm 18 years old, a first-time offender, and really want to study to become a nurse after high school, so I would really do anything to make up for this."
test_input_storyChap = "Hello, my name is Robert Chapman. I got caught speeding with my brand-new car the other day. I just had to see how fast it could go and oh man it was cool. However, I know it was wrong and I am sorry for the danger I inflicted on those around me. Then I honestly didn’t see the NIP come in my mail, as I never really read my mail. Now I guess that is what I really need to plead guilty for. And of course – if I had seen the notice, I would have probably responded to it. I'm 26 years old and already have penalty points from three previous speeding incidents on my licence. Other than the driving offenses, I have no record. I also have ADHD."
test_input_storyP = "Hey, I'm gonna need to plead guilty to being drunk and disorderly. I really needed a day of relaxing and not caring about everyone else for once. I do feel ashamed of my actions and wish I had better control of myself in that situation. I don’t want to do counselling, but I’ll do anything else you think would be appropriate to rectify this. My name is Paul Nicholls, I’m 54 years old, and I have one previous conviction of careless driving from 10 years ago."
test_input_storyL = "I am Lauren Dixon, 41 years old and want to plead guilty to TV licence payment evasion. I must stress that this was not intentional – I thought the bill was paid! But in reality, I had 8 months of unlicensed use. As soon as I realised my mistake, I paid for a licence properly. I have no previous criminal record, and I am ready to do whatever it takes to make this situation right."
test_input_storyChlo = "My name is Chloe Mills, I’m 24 years old, and pleading guilty to the offense of affray. I had just found out that my boyfriend cheated on me, so I went up to the bar I know he was at with his friends and threatened to get a gun and shoot him if he did not tell me the name of the other girl. Of course, I don’t own a gun or have any way of obtaining one, but he did not know that. I did slap him in the face but I did no more harm than that. I believe I was in the bar yelling at him for about 15 minutes before I went home. I am truly sorry for scaring everyone else in the bar at the time, and I realise I should have handled the situation in a better way. I have no criminal record and already go to therapy once a week to work on myself."
test_input_storyPalm = "Jack Palmer, 34 years old, I have had 1 speeding ticket, but my record is clean. I am guilty of being bribed, as I accepted a total of 4000 pounds in exchange for not publishing an article where I had found that this company emits one million tonnes of CO2 more per year than what they claim. I do not believe this has had much effect other than allowing the greenwashing of the company to continue for a few weeks because the true numbers were released by someone else just 5 weeks later."
test_input_storyNovak = "My name is Aleksander Nowak, I’d like to plead guilty to assault. I pushed my employee to the ground and then kicked him in the stomach. The reason behind this was that he had seriously messed up at work, so I got extremely angry with him, as I knew that his mistake would cost my company a lot of money. After it happened, I immediately regretted it and brought him to the closest emergency room to make sure he was ok. I truly regret this incident, and I want to apologise once more both to the victim and to his family. I have already offered to pay for the rest of his hospital bills. I am 40 years old and have no criminal record."
test_input_storyCharl = "I am pleading guilty to careless driving. I was on my way to the airport to pick up some family members, but I was running late. I was driving quite aggressively at a higher speed than what was permitted, and I answered a text message from my cousin while driving. The result of the situation was that I crashed into a guardrail. Luckily, no one else was hurt. I have 3 penalty points for speeding on my licence already. In hindsight, I have realised how dangerous my actions on this day were, and I deeply regret them. I know I could have caused harm not only to myself but to everyone around me on the road as well. I am 37 years old, I have an otherwise clean record, and I make regular donations to the British Heart Foundation. My name is Charlotte Porter."
test_input_storyLam = "My name is David Lambert, I am 55 years old and have no criminal record, other than the fact that I was disqualified from driving 2 months ago, due to four speeding incidents. I now plead guilty to driving whilst disqualified. I somehow didn’t think about the fact that I couldn’t drive, so I got in the car and drove to a supermarket. I am really embarrassed that I let this happen, and sorry that I didn’t act according to the court’s previous order. If I can somehow make amends or improve myself (take a course or something maybe), I’ll gladly do that."
#########################################################################################
# Train stories
input_storyR = "My name is Ronald Smith and I hit a guy. I am 32 years old and have never done anything like this. I had been drinking with some friends and then this guy said something very rude to my friend, so I got really angry and lost control for a moment."
input_storyM = "Hi, I’m Muhammad Hussain and unfortunately, I need to plead guilty for spitting in a man’s face. I was on my way home from work when I got some derogatory comments on the street. I was having a particularly bad day after losing a big client at work (I’m a lawyer), and so I got really angry. I talked back to the guy, but he kept nagging at me, and I ended up spitting in his face. I’m really sorry about it and I wish I could take it back. I would actually love to somehow gain better control of my anger by attending an anger management program. I am 32 years old, have never done anything criminal, and I am an eager member of the Muslim community in town, where I voluntarily help out about twice a week."
input_storyA = "I kicked a football through a storefront window. It was the middle of the day and I was walking home from soccer practice when I lost control of the ball and accidentally hit the window. There were a few people inside, but luckily no one was hurt. I am 19 years old and have never done anything criminal before. I feel so bad about this. And my name is Abbey Davies. I also have a really cute dog named Ollie."
input_storyW = "My name is Charlotte Wright and I am pleading guilty to drunk driving. My 8-year-old child was supposed to sleep over at a friend’s house but called me and wanted to go home after dinner. I had already had a few glasses of wine but did not consider that when getting in my car to go pick up my son. Frankly, I am really embarrassed by this incident, and incredibly remorseful, as I understand the severe consequences it could have had – not only for me and my son, but for others on the road as well. I already have 3 penalty points on my driver’s licence for speeding, but my son depends on me driving him to school every day, so I want to see if it is possible to not lose my licence. I’m 43 years old and don’t have anything else on my record."
input_storyJ = "My name is James Campbell. I run a small bike shop in town and I am guilty of tax evasion. My business has not been doing very well the past 2 years, and taxes became an expense I felt like I could not afford. So I didn’t register the work I did that was paid in cash, and did not provide receipts or put it in my books. I’m not sure of the exact amount, but I’m guessing I have failed to register about 7000 pounds. I also have a previous conviction of common assault. I do feel really bad about it and hope I will be able to make amends somehow. I have also decided to put down the business and apply for a steadier job to prevent anything like this from happening again.  I’m 48 years old."
#########################################################################################
#name = 'Lam'
name = sys.argv[1]
# python combined.py > g4/E/E2_

print(name)
if name=='O':
    input_story = test_input_storyO
elif name=='Mel':
    input_story = test_input_storyMel
elif name=='Chap':
    input_story = test_input_storyChap
elif name=='P':
    input_story = test_input_storyP
elif name=='L':
    input_story = test_input_storyL
elif name=='Chlo':
    input_story = test_input_storyChlo
elif name=='Palm':
    input_story = test_input_storyPalm
elif name=='Novak':
    input_story = test_input_storyNovak
elif name=='Charl':
    input_story = test_input_storyCharl
elif name=='Lam':
    input_story = test_input_storyLam
else:
    print("No valid name for test_stories, using another set.")
    input_story = input_storyM

system_prompt_statement = """Draft a statement for a plea in mitigation where the defendant will self-represent. The statement should follow a given set of principles, using information from the defendant's input story.\n"""
double_check = "Draft your answer using only information about the defendant and their case, from the input story. Do not make up more information to better satisfy the principles."
chat_log = [{"role": "system", "content": system_prompt_statement + principles},
            {"role": "user", "content": "### INPUT STORY: " + input_story},
            {"role": "user", "content": double_check + "\n### STATEMENT: "}]

initial_statement = ask_gpt(c_log=chat_log, temp=1)
toks = initial_statement["usage"]["total_tokens"]
initial_statement = initial_statement["choices"][0]["message"]["content"].strip()
print("initial statement:\n", toks)
print(initial_statement)
wait = True

# Loop until critique requests (CRs) are satisfied
for rep in range(1):
    retakes = 0
    principle_i = np.arange(0, len(yesnoCR))
    np.random.shuffle(principle_i)
    for i in principle_i:  # Which principles to test
        redoThis = 0
        while True:
            # Ask if current statement satisfies principle i
            print("\nCritique request: " + yesnoCR[i] +"\n" + yesnoWhy)
            diff = " Make sure to think carefully about what is said in the input story VS what is said in the statement."
            chat_log = [{"role": "system", "content": "You will be presented with a question, and then an input story and a statement. You must analyse the input story and statement in order to properly answer the question."},
                            {"role": "user", "content": "### QUESTION: " + yesnoCR[i] + yesnoWhy + diff},
                            {"role": "user", "content": "### INPUT STORY: " + input_story + "\n### STATEMENT: " + initial_statement + "\n### ANSWER:"}]
                
            
            critique = ask_g4(chat_log)["choices"][0]["message"]["content"].strip()

            critique = critique.strip().replace("\n", " ")
            print("g4 Critique: ", f"<{critique}>")

            if yesnoWhy != yesno_E:
                # Depending on the configuration of the fine-tune // input prompt, our actual answer may be placed at the beginning or end of the answer.
                # We now prioritize answers at the end of the critique. If there is no answer at the end, look for answer at the beginning of output.
                if "no" in critique.lower()[-10:]:
                    # Principle is satisfied. Move on to the next.
                    print("The statement satisfies the CR!")
                    bbreak = True
                elif "yes" in critique.lower()[-10:]:
                    # Principle is not satisfied. We want to rewrite the statement 
                    print("Principle NOT satisfied")
                    bbreak = False

                elif critique.lower()[:2] == "no":
                    # Principle is satisfied. Move on to the next.
                    print("The statement satisfies the CR!")
                    bbreak = True
                elif critique.lower()[:3] == "yes":
                    # Principle is not satisfied. We want to rewrite the statement 
                    bbreak = False
                else:
                    # Unwanted critique format, we could not find a proper answer.
                    # The behavior for this is currently to let it pass as "principle satisfied"
                    print("Unwanted format, we interpret as 'nothing wrong'")
                    bbreak = True
                
                if bbreak:
                    wait = False
                    break

                if redoThis >= 3:
                    # Stuck on claiming principle is still not satisfied...
                    print("Claims principle is not satisfied still. We plough through.")
                    break

                if wait:
                    # Free version of OpenAI needs to wait.
                    sl=True
                    #time.sleep(10)

            if yesnoWhy == yesno_A and not yesno_AR:
                # Ask GPT-3.5 turbo why current statement doesn't satisfy this principle
                system_prompt_critique = "Use information from the given input story to critique the below statement in a way that answers the critique request. " + diff + "\n" # Do NOT write a new statement.
                chat_log = [
                    {"role": "system", "content": system_prompt_critique + principles},
                    {"role": "system", "content": "### INPUT STORY: " + input_story + "\n### STATEMENT: " + initial_statement},
                    {"role": "user", "content": "### CRITIQUE REQUEST:\n" + whatCR[i] + "\n### CRITIQUE: "},
                    {"role": "assistant", "content": "Let's think step by step."}
                ]
                critique = ask_gpt(chat_log, temp=0.8, maxtoks=200)["choices"][0]["message"]["content"].strip().replace("\n", " ")
                print("\nExplanation request:", whatCR[i])
                print("\nCR not satisfied because:\n" + critique)


            # Rewrite the statement to better adhere to this principle
            no_hall = " Rewrite using only information about the defendant and their case from the input story. Do not make up information; even if you think that would improve the answer."
            if yesnoWhy == yesno_A and not yesno_AR:
                # If we have a 'simple critique'
                system_prompt_rewrite = "You will revise parts of a statement, based on a given critique and a revision request. Rewrite what is criticized (and quoted in the critique), and keep the rest as it was. You may want to delete a paragraph, or add a new one."
                chat_log = [
                    {"role": "system", "content": system_prompt_rewrite},
                    {"role": "system", "content": "### INPUT STORY: " + input_story + "\n### STATEMENT: " + initial_statement + "\n### CRITIQUE: " + critique},
                    {"role": "user", "content": "### REVISION REQUEST: " + rewriteR[i] + no_hall},
                    {"role": "user", "content": "### NEW STATEMENT: "}
                ]
            elif yesnoWhy == yesno_A and yesno_AR:
                system_prompt_rewrite = "You will revise parts of a statement, based on a given critique and a revision request. Rewrite what is criticized (and quoted in the critique), and keep the rest as it was. You may want to delete a paragraph, or add a new one."
                chat_log = [
                    {"role": "system", "content": system_prompt_rewrite},
                    {"role": "system", "content": "### INPUT STORY: " + input_story + "\n### STATEMENT: " + initial_statement},
                    {"role": "user", "content": "### REVISION REQUEST: " + rewriteR[i] + no_hall},
                    {"role": "user", "content": "### NEW STATEMENT: "}
                ]
            elif yesnoWhy == yesno_A and yesno_AR:
                system_prompt_rewrite = "You will revise parts of a statement, based on a given critique and a revision request. Rewrite what is criticized (and quoted in the critique), and keep the rest as it was. You may want to delete a paragraph, or add a new one."
                chat_log = [
                    {"role": "system", "content": system_prompt_rewrite},
                    {"role": "system", "content": "### INPUT STORY: " + input_story + "\n### STATEMENT: " + initial_statement},
                    {"role": "user", "content": "### REVISION REQUEST: " + rewriteR[i] + no_hall},
                    {"role": "user", "content": "### NEW STATEMENT: "}
                ]
            else: 
                # Include CR to give better context of the critique // explanation.
                system_prompt_rewrite = "You will revise parts of a statement, based on a given critique and revision request. If the critique points out a discrepancy between input story and statement, you should rewrite that part of the statement. The critique will quote or refer to parts of the statement that you need to decide to revise or not, based on the critique itself, and the revision request. Don't rewrite parts of the statement that are not criticized. You may want to delete a paragraph, or add a new one."
                chat_log = [
                    {"role": "system", "content": system_prompt_rewrite},
                    {"role": "system", "content": "### INPUT STORY: " + input_story + "\n### STATEMENT: " + initial_statement + "\n### CRITIQUE REQUEST: " + yesnoCR[i] +"\n### CRITIQUE: " + critique},
                    {"role": "user", "content": "### REVISION REQUEST: " + rewriteR[i] + no_hall},
                    {"role": "user", "content": "### NEW STATEMENT: "}
                ]

            new_statement = ask_gpt(chat_log, temp=1, pres_pen=0.3)            
            print("NEW STATEMENT:\n")
            print(new_statement["usage"]["total_tokens"])
            #print(new_statement["choices"][0]["message"]["content"].strip())
            initial_statement = new_statement["choices"][0]["message"]["content"].strip()
            # New loop to see if the new statement satisfies the CR.
            print(initial_statement)
            retakes += 1
            redoThis += 1
            wait = True

            if yesnoWhy == yesno_E:
                # Jump to the next principle; don't revise this principle forever
                break

    print(f"There were {retakes} yes-es this loop.")
    if retakes == 0:
        break 

    print("That was one loop through all principles!")
#print("The statement now satisfies all principles:\n")
print(initial_statement)