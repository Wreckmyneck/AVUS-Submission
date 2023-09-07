"""
This code turned out to not give sufficient results to be further attempted/improved upon within the time-frame.
Due to the time-frame the code is also unrefined and lacking comment.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from tabulate import tabulate
from tqdm import trange
import random

file_path = r"C:\Users\Conor\Desktop\Summer Code\textfiles\Merged\training_dataset_full.csv"
df = pd.DataFrame({'label':int(), 'text':str()}, index = [])
whole_text = []
with open(file_path, encoding="utf-8") as f:
        for line in f.readlines():
            split = line.split('\t')
            whole_text.append(split)

whole_text = whole_text[1:]
label_content_pairs = [(item[0].strip(), item[1].strip()) for item in whole_text]

# Convert list of tuples into a pandas DataFrame
df = pd.DataFrame(label_content_pairs, columns=['label', 'content'])

# Display the resulting DataFrame
#print(df)
text = df.content.values
labels = df.label.astype(int).values
#print(text)
#print(labels)

#process text
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
    )


def print_rand_sentence():
  '''Displays the tokens and respective IDs of a random text sample'''
  index = random.randint(0, len(text)-1)
  table = np.array([tokenizer.tokenize(text[index]), 
                    tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text[index]))]).T
  print(tabulate(table,
                 headers = ['Tokens', 'Token IDs'],
                 tablefmt = 'fancy_grid'))

print_rand_sentence()

token_id = []
attention_masks = []

def preprocessing(input_text, tokenizer):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''

  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 32,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation=True
                   )


for sample in text:
  encoding_dict = preprocessing(sample, tokenizer)
  token_id.append(encoding_dict['input_ids']) 
  attention_masks.append(encoding_dict['attention_mask'])


token_id = torch.cat(token_id, dim = 0)
attention_masks = torch.cat(attention_masks, dim = 0)
labels = torch.tensor(labels)

def print_rand_sentence_encoding():
  '''Displays tokens, token IDs and attention mask of a random text sample'''
  index = random.randint(0, len(text) - 1)
  tokens = tokenizer.tokenize(tokenizer.decode(token_id[index]))
  token_ids = [i.numpy() for i in token_id[index]]
  attention = [i.numpy() for i in attention_masks[index]]

  table = np.array([tokens, token_ids, attention]).T
  print(tabulate(table, 
                 headers = ['Tokens', 'Token IDs', 'Attention Mask'],
                 tablefmt = 'fancy_grid'))

print_rand_sentence_encoding()

val_ratio = 0.2
# Recommended batch size: 16, 32.
batch_size = 16
# Indices of the train and validation splits stratified by labels
train_idx, val_idx = train_test_split(
    np.arange(len(labels)),
    test_size = val_ratio,
    shuffle = True,
    stratify = labels)
# Train and validation sets
train_set = TensorDataset(token_id[train_idx], 
                          attention_masks[train_idx], 
                          labels[train_idx])

val_set = TensorDataset(token_id[val_idx], 
                        attention_masks[val_idx], 
                        labels[val_idx])

# Prepare DataLoader
train_dataloader = DataLoader(
            train_set,
            sampler = RandomSampler(train_set),
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            val_set,
            sampler = SequentialSampler(val_set),
            batch_size = batch_size
        )

def b_tp(preds, labels):
  '''Returns True Positives (TP): count of correct predictions of actual class 1'''
  return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_fp(preds, labels):
  '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
  return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_tn(preds, labels):
  '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
  return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_fn(preds, labels):
  '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
  return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_metrics(preds, labels):
  '''
  Returns the following metrics:
    - accuracy    = (TP + TN) / N
    - precision   = TP / (TP + FP)
    - recall      = TP / (TP + FN)
    - specificity = TN / (TN + FP)
  '''
  preds = np.argmax(preds, axis = 1).flatten()
  labels = labels.flatten()
  tp = b_tp(preds, labels)
  tn = b_tn(preds, labels)
  fp = b_fp(preds, labels)
  fn = b_fn(preds, labels)
  b_accuracy = (tp + tn) / len(labels)
  b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
  b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
  b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
  return b_accuracy, b_precision, b_recall, b_specificity

# Load the BertForSequenceClassification model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)

# Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr = 5e-5,
                              eps = 1e-08
                              )

model.cuda()

device = torch.device('cpu')
model.to(device)

# Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
epochs = 2

for _ in trange(epochs, desc = 'Epoch'):
    
    # ========== Training ==========
    
    # Set model to training mode
    model.train()
    
    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        b_labels = b_labels.to(torch.long)
        optimizer.zero_grad()
        # Forward pass
        train_output = model(b_input_ids, 
                             token_type_ids = None, 
                             attention_mask = b_input_mask, 
                             labels = b_labels)
        # Backward pass
        train_output.loss.backward()
        optimizer.step()
        # Update tracking variables
        tr_loss += train_output.loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    # ========== Validation ==========

    # Set model to evaluation mode
    model.eval()

    # Tracking variables 
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_specificity = []

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            # Forward pass
            eval_output = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
        logits = eval_output.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy().astype(np.int64)  # Convert to NumPy int64
        # Calculate validation metrics
        b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
        val_accuracy.append(b_accuracy)
        # Update precision only when (tp + fp) != 0; ignore nan
        if b_precision != 'nan':
            val_precision.append(b_precision)
        # Update recall only when (tp + fn) != 0; ignore nan
        if b_recall != 'nan':
            val_recall.append(b_recall)
        # Update specificity only when (tn + fp) != 0; ignore nan
        if b_specificity != 'nan':
            val_specificity.append(b_specificity)

    print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
    print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
    print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
    print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
    print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')


    new_sentences = ["""Ukrainian forces have made an attempt to cross the Dnipro River dividing liberated and occupied Kherson, in a potential breach of what has for months served as the frontline in the south of Ukraine.

Russian military bloggers reported that up to seven boats, each carrying six to seven people, had landed near the village of Kozachi Laheri, east of Kherson city, on Tuesday and broke through Russian defensive lines.

It was claimed that the Ukrainian soldiers had advanced up to 800 metres after getting to the riverbank, though it appeared Russian forces had some success in fighting them back.

The Russian-imposed head of the occupied part of the Kherson oblast, Vladimir Saldo, claimed the Ukrainian raid had been repelled.

The respected Institute for the Study of War (ISW) thinktank in Washington said, however, that it appeared a “limited raid” may have had more success than Saldo had acknowledged.

In the latest update, the ISW wrote: “The majority of prominent Russian [military bloggers] claimed that Ukrainian forces managed to utilise tactical surprise and land on the east bank before engaging Russian forces in small arms exchanges, and Saldo was likely purposefully trying to refute claims of Ukrainian presence in this area to avoid creating panic in the already-delicate Russian information space.”

It added that there was satellite imagery to suggest there had been a major battle in the area.

“Hotspots on available Nasa Fire Information for Resource Management System (Firms) data from the past 24 hours in this area appear to confirm that there was significant combat, likely preceded or accompanied by artillery fire,” the ISW said. “By the end of the day on 8 August, many Russian sources had updated their claims to report that Russian forces retain control over Kozachi Laheri, having pushed Ukrainian forces back to the shoreline, and that small arms skirmishes are occurring in shoreline areas near Kozachi Laheri and other east bank settlements.”

There have been a number of attempts by Ukrainian forces to cross the Dnipro River, which has been established as the dividing line between the warring nations since Ukraine’s successful offensive in Kherson in November pushed on to what is known as the “left bank” of the river.

In June, a raid was launched by Ukraine’s elite 73rd marine special operations unit, but the latest landing appears to have been the most significant of recent months despite doubts over the sustainability of the Ukrainian positions.

In Moscow, there were reports that at least 45 people had been injured in an explosion at the Zagorsk optical and mechanical plant in Sergiev Posad, near the capital, up from an earlier figure of at least 30 people. The cause of the incident is unclear.""",
"""At least 41 people are feared to have died after a boat sank in rough seas off the Italian island of Lampedusa, in the central Mediterranean, according to media reports.

Four survivors who were rescued on Wednesday morning by a Maltese bulk carrier, and eventually moved to a patrol boat from the Italian coastguard, said they were on a vessel that had set off from Sfax, in Tunisia, and sank on its way to Italy’s shores.

The asylum seekers, three men and a woman from Ivory Coast and Guinea, said their vessel, a precarious metal boat carrying 45 passengers, including three children, had begun to take on water as soon as they reached the open sea.

“Suddenly we were overwhelmed by a giant wave,” one survivor told the coastguard.

Almost all the passengers, who are believed to be from sub-Saharan Africa, ended up in the open, stormy sea for hours. According to the testimonies of the four, at least 41 passengers are believed to have drowned.

Neither the Maltese bulk carrier nor the Italian coastguard boat came across any of the victims’ bodies.""",
"""Mr. Milton Smith, Executive Vice President in Charge of Production, was in conference. A half dozen men lounged comfortably in deep, soft chairs and divans about his large, well-appointed office in the B.O. studio. Mr. Smith had a chair behind a big desk, but he seldom occupied it. He was an imaginative, dramatic, dynamic person. He required freedom and space in which to express himself. His large chair was too small; so he paced about the office more often than he occupied his chair, and his hands interpreted his thoughts quite as fluently as did his tongue.

"It's bound to be a knock-out," he assured his listeners; "no synthetic jungle, no faked sound effects, no toothless old lions that every picture fan in the U. S. knows by their first names. No, sir! This will be the real thing."

A secretary entered the room and closed the door behind her. "Mr. Orman is here," she said.

"Good! Ask him to come in, please." Mr. Smith rubbed his palms together and turned to the others. "Thinking of Orman was nothing less than an inspiration," he exclaimed. "He's just the man to make this picture."

"Just another one of your inspirations, Chief," remarked one of the men. "They've got to hand it to you."

Another, sitting next to the speaker, leaned closer to him. "I thought you suggested Orman the other day," he whispered.

"I did," said the first man out of the corner of his mouth.

Again the door opened, and the secretary ushered in a stocky, bronzed man who was greeted familiarly by all in the room. Smith advanced and shook hands with him.

"Glad to see you, Tom," he said. "Haven't seen you since you got back from Borneo. Great stuff you got down there. But I've got something bigger still on the fire for you. You know the clean-up Superlative Pictures made with their last jungle picture?"

"How could I help it; it's all I've heard since I got back. Now I suppose everybody's goin' to make jungle pictures."

"Well, there are jungle pictures and jungle pictures. We're going to make a real one. Every scene in that Superlative picture was shot inside a radius of twenty-five miles from Hollywood except a few African stock shots, and the sound effects—lousy!" Smith grimaced his contempt.

"And where are we goin' to shoot?" inquired Orman; "fifty miles from Hollywood?"

"No, sir! We're goin' to send a company right to the heart of Africa, right to the—ah—er—what's the name of that forest, Joe?"

"The Ituri Forest."

"Yes, right to the Ituri Forest with sound equipment and everything. Think of it, Tom! You get the real stuff, the real natives, the jungle, the animals, the sounds. You 'shoot' a giraffe, and at the same time you record the actual sound of his voice."

"You won't need much sound equipment for that, Milt."

"Why?"

"Giraffes don't make any sounds; they're supposed not to have any vocal organs."

"Well, what of it? That was just an illustration. But take the other animals for instance; lions, elephants, tigers—Joe's written in a great tiger sequence. It's goin' to yank 'em right out of their seats."

"There ain't any tigers in Africa, Milt," explained the director.

"Who says there ain't?"

"I do," replied Orman, grinning.

"How about it, Joe?" Smith turned toward the scenarist.

"Well, Chief, you said you wanted a tiger sequence."

"Oh, what's the difference? We'll make it a crocodile sequence."

"And you want me to direct the picture?" asked Orman.

"Yes, and it will make you famous."

"I don't know about that, but I'm game—I ain't ever been to Africa. Is it feasible to get sound trucks into Central Africa?"

"We're just having a conference to discuss the whole matter," replied Smith. "We've asked Major White to sit in. I guess you men haven't met—Mr. Orman, Major White," and as the two men shook hands Smith continued. "The major's a famous big game hunter, knows Africa like a book. He's to be technical advisor and go along with you."

"What do you think, Major, about our being able to get sound trucks into the Ituri Forest?" asked Orman.

"What'll they weigh? I doubt that you can get anything across Africa that weighs over a ton and a half."

"Ouch!" exclaimed Clarence Noice, the sound director. "Our sound trucks weigh seven tons, and we're planning on taking two of them."

"It just can't be done," said the major.

"And how about the generator truck?" demanded Noice. "It weighs nine tons."

The major threw up his hands. "Really, gentlemen, it's preposterous."

"Can you do it, Tom?" demanded Smith, and without waiting for a reply. "You've got to do it."

"Sure I'll do it—if you want to foot the bills."

"Good!" exclaimed Smith. "Now that's settled let me tell you something about the story. Joe's written a great story—it's goin' to be a knock-out. You see this fellow's born in the jungle and brought up by a lioness. He pals around with the lions all his life—doesn't know any other friends. The lion is king of beasts; when the boy grows up he's king of the lions; so he bosses the whole menagerie. See? Big shot of the jungle.""",
"""All fagged out, I dragged myself wearily from the sun-baked concrete highway to the skinny shade of a thin-limbed, thirsty-looking bush.

“Under the spreading blacksmith tree the village chestnut sits,” I crazily recited, kicking off my shoes to cool my blistered feet. Then I looked at my chum with begging eyes. “Get me some ice cream, Poppy. Quick, before I faint.”

Boy, was I ever hot! I felt like a fried egg. But scorched as I was, inside and out, I could still sing a song.

To better introduce myself, I’ll explain that my name is Jerry Todd. I live in Tutter, Illinois, which is the peachiest small town in the state. And the kids I run around with are the peachiest boy pals in the state, too, particularly Poppy Ott, the hero of this crazy story.

[2]Poppy is a real guy, let me whisper to you. I never expect to have a chum whom I like any better than I like him. He’s full of fun, just like his funny name, which he got from peddling pop corn. And brains? Say, when they were dishing out gray matter old Poppy got served at both ends of the line. I’ll tell the world. If you want to know how smart he is, just read POPPY OTT’S SEVEN-LEAGUE STILTS. Starting with nothing except an idea, we ended up, under his clever leadership, with a factory full of stilt-manufacturing machinery and money in the bank. That’s Poppy for you. Every time. A lot of his ideas are pretty big for a boy, but he makes them work. Of course, as he warmly admits, I was a big help to him in putting the new stilt business on its feet and teaching it to stand alone. But his loyal praise doesn’t puff me up. For I know who did the most of the headwork.

With Poppy’s pa doing the general-manager stuff in the new factory, my chum and I had merrily set forth on a hitch-hike as a sort of vacation. This, too, was Poppy’s idea. A hitch-hike, as every kid knows, is a sort of free automobile tour. You start walking down the concrete in the direction you want to go, and when a motor car to your liking comes alone you wigwag the driver to stop and give you a lift. Sometimes you get it and sometimes you don’t.[3] But if you limp a little bit, and act tired, that helps.

Poppy, of course, was all hip-hip-hurray over his hitch-hike idea. That’s his way. Our most violent exercise, he spread around, seeing nothing but joy and sugar buns ahead, would be lifting our travel-weary frames into soft-cushioned Cadillacs and Packards. Once comfortably seated, we would glide along swiftly and inexpensively. No gasoline bills to pay. No new tires to buy. Everything free, including the scenery. Some automobiles would carry us ten miles, others would carry us a hundred miles. “We might even average around three hundred miles a day,” was some more of his line, “and still have time each night to stop at a farmhouse and do chores for our supper and breakfast.” If we slept in the farmer’s barn, that would be free, too. Our trip would cost us scarcely anything, though it would be wise, the leader tacked on at the tail end, to carry twenty dollars in small bills for emergencies.

I fell for the scheme, of course. For Poppy never has any trouble getting me to do what he wants me to do. Not that I haven’t a mind of my own. But I’ve found out that in going along with him I usually learn something worth while, and have a whale of a lot of fun doing it, too.

Having won our parents’ consent to the trip, we had set forth that morning in high feather. But in[4] poor luck we now were held up on a closed road, though why the road had been suddenly shut off was a mystery to us.

With a final look up and down the long stretch of concrete, Poppy came over to where I was and dropped down beside me in the hot sand.

“Still not a sign of a car,” says he.

“Not even a flivver, huh?” I suffered with him.

“I can’t understand it,” says he, puzzled. “We saw a few cars after we left Pardyville. But the road’s completely empty now, and has been for hours.”

I saw a chance to have some fun with him.

“‘And our most violent exercise,’” I quoted glibly, “‘will be lifting our travel-weary frames into soft-cushioned cattle racks and pant hards.’ Say, Poppy,” I grinned, “was that last cattle rack we rode in a four-legged wheelbarrow or another gnash?”

“You won’t feel so funny,” came the laugh, “if you have to go to bed to-night without your supper.”

“Bed?” says I, looking around at the sun-baked scenery. It was a beautiful country, all right—for sand burs and grasshoppers! “Where’s the bed?” I yawned. “Lead me to it.”

“This sand knoll may be the only bed you’ll get. For there isn’t a farmhouse in sight.”

I got my eyes on something.

[5]“The Hotel Emporia for me, kid,” I laughed, pointing to a billboard beside the highway. “‘One hundred comfortable rooms,’” I read, “‘each with bath and running ice water. Delectable chicken dinners. Sun-room cafeteria. Inexpensive garage in connection.’ Who could ask for more?” I wound up.""",
"""INDIA BREAKS THE CHAINS OF COLONIAL RULE: CELEBRATES INDEPENDENCE DAY

New Delhi, August 15, 1947 - A historic moment unfolds as India, after centuries of British colonial rule, declares its independence and emerges as a sovereign nation. The long-awaited freedom marks the culmination of tireless struggle, sacrifice, and determination by millions of Indians who fought for their rights and the dream of self-governance.

The Struggle for Independence

The journey to independence has been arduous, spanning decades of organized resistance against British colonialism. From the Non-Cooperation Movement, spearheaded by Mahatma Gandhi, to the Quit India Movement, India witnessed the unyielding spirit of its people standing together against foreign dominion.

The peaceful resistance, civil disobedience, and mass protests held under the banner of unity echoed across the country and resonated globally. The world stood witness to India's relentless pursuit of freedom and its unwavering commitment to securing its own destiny.

The Midnight Tryst with Destiny

At the stroke of midnight, as the 14th of August gave way to the 15th, the Indian flag was hoisted, and the first Prime Minister of India, Pandit Jawaharlal Nehru, addressed the nation with his iconic "Tryst with Destiny" speech. In his stirring words, he spoke of India's commitment to justice, liberty, and equality, and the responsibility that lay ahead to shape the nation's destiny.

A New Dawn Rises

Today, as India celebrates its first Independence Day, the atmosphere across the country is electric. Citizens have taken to the streets, waving the tricolor flag, and participating in parades and festivities to commemorate this historic occasion.

The Birth of a Nation

The nation of India is born as the result of years of dedication and the collective spirit of its people. The Indian National Congress and other political parties, alongside religious and social groups, have united to form a nation built on democratic principles and secular values.

Unity in Diversity

India, known for its diversity of languages, religions, cultures, and traditions, now stands united under the umbrella of independence. The vision of a nation where all citizens enjoy equal rights and opportunities is one that will be tirelessly pursued by its leaders and citizens alike.

Challenges Ahead

As India embraces its independence, the challenges that lie ahead are significant. The task of nation-building, economic development, and fostering social cohesion will require dedication, cooperation, and resilience from all corners of society.

A Bright Future

Amidst the challenges, India's spirit remains undeterred. The resilience that has guided the nation through the struggle for independence will serve as the foundation for a brighter future. The dreams of a prosperous, inclusive, and progressive India are within reach, and the nation is poised to seize its opportunities and confront its challenges head-on.

A Message to the World

India's independence is not just a milestone for the nation but a message to the world. The power of unity, non-violence, and perseverance has demonstrated that when a people stand together, they can break the shackles of oppression and forge a new path towards freedom and self-determination.

As the Indian tricolor flutters high above the Red Fort in Delhi, the echoes of "Jai Hind" resound across the land. Today, India stands tall and proud, ready to carve its unique place in the world, with a promise to cherish its hard-earned independence and uphold the values of justice, equality, and fraternity.""",
"""Historic Good Friday Agreement Brings Hope for Lasting Peace in Northern Ireland

Belfast, Northern Ireland - In a momentous occasion that will be etched into the history of Northern Ireland, leaders from across the political spectrum have come together to sign the Good Friday Agreement, a landmark accord aimed at bringing an end to decades of sectarian violence and division.

A New Dawn of Peace

The Good Friday Agreement, also known as the Belfast Agreement, is the result of years of painstaking negotiations and compromises between representatives from the British and Irish governments, as well as the main political parties of Northern Ireland. The agreement, signed today at Stormont Castle, marks a significant step towards lasting peace and stability in the region.

Key Provisions of the Agreement

The Good Friday Agreement encompasses a range of critical provisions addressing the core issues that have fueled the conflict in Northern Ireland. Some key aspects of the agreement include:

    Power-sharing Assembly: The establishment of a new Northern Ireland Assembly, wherein elected representatives from both the nationalist and unionist communities will participate in governance, allowing all voices to be heard and represented.

    Decommissioning of Weapons: A crucial commitment to the decommissioning of paramilitary weapons, providing a path for armed groups to lay down their arms and embrace peaceful politics.

    Human Rights and Equality: A strong emphasis on human rights, equality, and mutual respect, aiming to build a society where everyone's rights are protected, irrespective of their background.

    North-South Cooperation: Enhanced cooperation and interaction between Northern Ireland and the Republic of Ireland, strengthening cross-border ties and promoting economic and social development.

    Prisoner Release: A provision for the release of prisoners affiliated with paramilitary groups, subject to certain conditions and timelines.

Global Acclaim

The Good Friday Agreement has garnered widespread international acclaim, with leaders from across the world praising the courage and determination of all parties involved in reaching this historic milestone. The United States, the European Union, and other nations have offered their support and commitment to assist in the implementation of the agreement.

Challenges Ahead

Despite the euphoria surrounding the signing of the agreement, everyone acknowledges that the road to lasting peace will not be without challenges. The process of decommissioning weapons and rebuilding trust among communities will require time and patience. Furthermore, there may be elements on both sides who seek to derail the progress. However, the commitment demonstrated by leaders today instills hope that these hurdles can be overcome.

A Collective Responsibility

In the wake of the Good Friday Agreement, the responsibility to ensure its success now rests on the shoulders of the people of Northern Ireland, as well as their elected representatives. The agreement offers a unique opportunity to forge a new path, leaving behind the legacy of violence and division, and embracing a future characterized by cooperation, dialogue, and shared prosperity.

The Dawn of a New Era

As the ink dries on the historic document, the people of Northern Ireland look forward to a new era—one defined by peace, reconciliation, and unity. The Good Friday Agreement stands as a testament to the resilience of all those involved in the peace process, and today will be remembered as a day of hope for a brighter future for this beautiful land."""]

test_ids = []
test_attention_mask = []
total_encodings = []

for strings in new_sentences:
   encodings = preprocessing(strings, tokenizer)
   total_encodings.append(encodings)

index = 0
for encoding in total_encodings:
    test_ids = []
    test_attention_mask = []
    test_ids.append(encoding['input_ids'])
    test_attention_mask.append(encoding['attention_mask'])
    test_ids = torch.cat(test_ids, dim = 0)
    test_attention_mask = torch.cat(test_attention_mask, dim = 0)

    # Forward pass, calculate logit predictions
    with torch.no_grad():
        output = model(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))

    prediction = 'Human Written' if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else 'AI-generated'

    print('Input Sentence: ', new_sentences[index])
    print('Predicted Class: ', prediction)
    index = index+1