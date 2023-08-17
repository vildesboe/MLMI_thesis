from key import test_key, miri_key
import openai
import time

openai.api_key = miri_key

def auto_eval_4(story, statement):
    evaluate_assertions = "Below you will see a list of general assertions followed by a statement, and an input story. Consider the input story and statement, and go through one by one assertion, discussing if this statement and input story pair holds or breaks that assertion. Output this as a list in the same order as the assertions."
    assertions = "- The statement is well-written, non-repetitive, and easy to follow.\n- The statement mentions that the defendant is pleading guilty.\n- The statement does not contain any completely irrelevant information (eg it does not mention the 'input story' or another statement).\n- The statement only says that the defendant has suffered, if this is stated by the input story. If the input story implies that the defendant has suffered as a consequence of the offense, the statement should include this. Great emotional suffering is not mentioned in the statement.\n- If the defendant's age is below 30, or above 79, it is included in the statement. If the defendant's age is between 30-79, the age is not mentioned in the statement.\n- If the input story implies that the defendant has a clean record, this is mentioned in the statement. The statement does imply that the defendant has a clean record, if the input story cannot support this.\n- If the input story states that the defendant has a mental illness or disability, this is included in the statement. If the input story says nothing about a mental illness or disability, no such thing is mentioned in the statement either.\n- The statement properly explains what the offense is and how it happened - as in the input story. It does not add or skip important parts from the input story.\n- If the input story states that the defendant wants to or has already made amends; this is included in the statement (including examples, if this is present in the input story). If amends (eg compensating the victim) is not mentioned in the input story, it is not mentioned in the statement.\n- If the input story states that the defendant wants to or has already made efforts to self-improve or prevent the offense from happening again; this is included in the statement (including examples, if this is present in the input story). If the input story does not mention self-improvement (eg counselling) or preventing repetition of the crime, it is not mentioned in the statement either.\n- If the input story gives information that speaks to the defendant's good character, the statement includes this (including examples, if this is present in the input story). If no character-building information (eg voluntary work or contributing to society) is given in the input story, it is not mentioned in the statement either.\n- The emotions in the statement is the same as the emotions stated in the input story. Eg the statement does not show more regret, guilt or anger than the input story.\n- All information found in the statement is supported by information in the input story."
    #get_penalties = "Based on these evaluations - how many assertions are broken for our input story and statement pair?"
    diff = "Make sure to think carefully about what is said in the input story VS what is said in the statement."

    chat_log = [{"role": "system", "content": evaluate_assertions + diff},
                {"role": "user", "content": assertions + '\n' + "### INPUT STORY: " + story + '\n' + "\n### STATEMENT: " + statement}]

    try:
        answ = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=chat_log,
            temperature=0.1,
            top_p = 0.9,
        )['choices'][0]['message']['content'].strip()
    except:
        time.sleep(10)
        answ = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=chat_log,
            temperature=0.1,
            top_p = 0.9,
        )['choices'][0]['message']['content'].strip()


    def count_breaks(answer):
        nr_to_princ = {0: 'coherence', 1:'guilty plea', 2:'irrelevant', 3:'suffering', 4:'age', 5:'record', 6:'mental', 7:'explain', 8:'amends', 9:'improve + prevent', 10:'character', 11:'emotions', 12:'hallucination'}
        per = answer.strip('- ').split("\n- ")
        broken = 0
        alle = 0
        still_none = []
        broken_ones = []
        for t, j in enumerate(per):
            if "True" in j and "False" in j:
                print("BOTH", j)
            elif 'False' in j:
                broken += 1
                broken_ones.append(t)
                #print("broken princ:", nr_to_princ[t])
            elif not ("True" in j or "False" in j):
                #print(t, "none")
                if ': Yes' in j and ': No' in j:
                    print('both Y and N')
                elif ': Yes' in j:
                    continue
                elif ': No' in j:
                    broken += 1
                    broken_ones.append(t)
                    #print("Broken YN: ", nr_to_princ[t])
                elif not (": Yes" in j or ": No" in j):
                    if ": The statement holds this assertion" in j or ": This assertion holds" in j or "The claim is satisfied." in j:
                        continue
                    elif ": The statement breaks this assertion" in j or ": This assertion does not hold" in j or "The claim is not satisfied" in j or ": This assertion breaks" in j:
                        broken += 1
                        broken_ones.append(t)
                        #print("Broken YN: ", nr_to_princ[t])
                    else:
                        if ": Breaks" in j or 'This breaks the assertion.' in j:
                            broken += 1
                            broken_ones.append(t)
                            #print("Broken: ", nr_to_princ[t])
                        elif ": Holds" in j:
                            continue
                        else:
                            print("Still none:",  nr_to_princ[t])
                            still_none.append(t)

        if len(still_none)>0:
            for p in still_none:
                part = answer.strip('- ').split("\n- ")[p]
                #print(part)

                chat_log = [{"role": "system", "content": "Below you will see a claim, and then a discussion of the claim. The structure of this is as follows \nclaim: discussion\nBased on the discussion, would you say that the claim is satisfied or not?"},
                            {"role": "user", "content": part}]

                response = openai.ChatCompletion.create(
                    #model="gpt-3.5-turbo-0613",
                    model="gpt-4-0613",
                    messages=chat_log,
                    temperature=0.1,
                    top_p = 0.9,
                    )['choices'][0]['message']['content'].strip()

                print("New explanation for princ", nr_to_princ[p], ": ", response)
                if "not satisfied" in response.lower().strip() or response.lower().strip()=="no":
                    broken += 1
                    broken_ones.append(p)
                    print("Broken by 2nd check: ", nr_to_princ[p])
                    print("Original answer: ", part)
                else:
                    print("Princ not considered broken.", nr_to_princ[p])
                    print("Original answer: ", part)

                #print(f"# broken principles: {len(b_principles)}/13")

        return  [nr_to_princ[i] for i in broken_ones]
    
    return count_breaks(answ)

if __name__ == "__main__":
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
    
    # ---------------------------------------------------------
    # Change this part
    out_files = ["Chap", "Charl", "Chlo", "L", "Lam", "Mel", "Novak", "O", "P", "Palm"]#
    comments = [""]
    input_stories = [test_input_storyChap, test_input_storyCharl, test_input_storyChlo, test_input_storyL, test_input_storyLam, test_input_storyMel, test_input_storyNovak, test_input_storyO, test_input_storyP, test_input_storyPalm]

    pre_name = 'logs/oob_vicuna/A/A5_'
    pre_name = 'logs/finetunes_new/Vicuna/A/A5_'
    #pre_name = 'g4/A/A5_'
    #pre_name = 'DR/direct5_'
    end_name = ''
    # ---------------------------------------------------------
    out_files = [o+end_name for o in out_files]

    import numpy as np
    penalties = np.array([])
    print(f"#We have {len(out_files)} files")
    for i in range(len(out_files)):
        out_file = out_files[i]
        input_story = input_stories[i]

        file = open(pre_name+out_file,mode='r')
        statement = file.read().split("yes-es this loop.")[-1].split("one loop through all principles!")[-1].strip()
        file.close()
        statement = statement.replace("\n\n", " ").replace("\n", " ")

        print(pre_name, out_file)#, statement)
        ev = auto_eval_4(input_story, statement)
        print(f"{ev}: {len(ev)}")
        print()
        penalties = np.append(penalties, len(ev))

    print(f"#Average nr penalties: {np.mean(penalties)}")