# Welcome to SentimentalDestroR
This repository focuses on the sentiment classification library [**pysentimiento**](https://github.com/pysentimiento/pysentimiento) and **[SiEBERT](https://huggingface.co/siebert/sentiment-roberta-large-english)**. We intend to demonstrate how the performance of the classifier reduces due to the introduction of augmented sentences.
For the augmentation we use the augmentation recipe EDA, [Easy Data Augmentation](https://arxiv.org/abs/1901.11196) provided by [**TextAttack**](https://textattack.readthedocs.io/en/latest/3recipes/augmenter_recipes.html). As this augmentation recipe provides the best results for variance testing. It uses WordNet synonym replacement, Word deletion, Word order swaps, and, Random synonym insertion, at the same time.
 
# Dataset Description
:open_file_folder: Sentiment Dataset 
For this demonstration we focus on the dataset of [SemEval17 Task 4](https://alt.qcri.org/semeval2017/task4/index.php?id=data-and-tools), [IMBD](https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis), and [Rotten Tomatoes](https://huggingface.co/datasets/rotten_tomatoes) which is a very popular dataset for sentiment analysis. 
Folder structure:
```
Sentiment_Dataset
├── IMDB
│   ├── Augmented
│   │   └── log.csv
│   ├── Clean
│   │   └── movie.csv
│   └── Raw
│       ├── archive
│       │   └── movie.csv // actual raw dataset
│       └── archive.zip
├── SemEval17
│   ├── Augmented
│   │   ├── augmented(20%)_SemEval17.csv
│   │   ├── augmented_SemEval17.csv
│   │   └── predicted_semeval17.csv
│   ├── augmented_SemEval17.csv
│   ├── Clean
│   │   └── semeval.csv
│   └── Raw
│       ├── 4a-english.zip
│       ├── baseline-A-english.txt
│       ├── README.txt
│       ├── SemEval2017-task4-dev.subtask-A.english.INPUT.txt // actual raw dataset
│       ├── SemEval2017_task4_test_scorer_subtaskA.pl
│       └── twitter-2016test-A-English.txt
└── Sentiment140
    ├── Augmented
    │   ├── augmented_sentiment140.csv
    │   └── predicted_sentiment140.csv
    ├── Clean
    │   ├── clean_sentiment140.csv
    │   └── sentiment140.csv
    └── Raw
        └── training.1600000.processed.noemoticon.csv
```


## Get the dataset Dataset
First run the dataset.py file to create the dataset folder(Sentiment_Dataset). Or download it from the [link](https://drive.google.com/drive/u/1/folders/1BjzJC6voM8KP3OxyD2LXXkvgAoataW1a). 
:warning: Still under development. Please contact the developer.
```
    python dataset.py [-h]          Provides documentation
                      [--help]
                      
                      [--create_database] Creates the "Databases" directory 
                                          in the root directory with all the datasets
``` 

## File Description
|	File Name	|	Usage	|
|--|--|
| :snake: augmentation_SemEval17.ipynb  | Creates the 50% augmented dataset of the SemEval17 Dataset and evaluates it(raw) |
| :snake: augmentation_SemEval17(20%).ipynb | Creates the 20% augmented dataset of the SemEval17 Dataset and evaluates it(raw) |
| :snake: augmentation_sentiment140.ipynb | :warning: Creates augmented dataset for Sentiment140 |
| :snake: TextattackOnHuggingFaceRottenTomatoes.ipynb | Creates EDA augmentations over Rotten Tomatoes dataset(CLI) |
| :snake: TextattackOnIMDB.ipynb | Creates EDA augmentations over IMDB dataset(wrapper) |


:warning: The Sentiment140 Dataset is unusable due to the discrepancy polarity of the dataset and the classifier model. 


## How the pipeline works
```mermaid
graph LR
A[Create git clone of the repo] --> B((2 Virtual Env <br> Creation))
B-- install from venv_txtatck_req.txt --> D[venv_txtatck]
B-- install from venv_pysentimiento_req.txt --> C[venv_pysentimiento] 
D -- use it for augmentation purpose --> E[AugmentedDataset]
C -- use it for prediction & evaluation purpose --> E[Augmented Dataset]
E -- generate the result report --> F[Results]
```
## Result 
We can notice significant reduction on the performance of the model for the introduction of the augmented dataset
###  Raw Augmentation on Pysentimiento
#### Performance Before Augmentation
```python
REPORT
              precision    recall  f1-score   support

         NEG       0.79      0.88      0.83      3231
         NEU       0.88      0.82      0.85     10342
         POS       0.84      0.88      0.86      7059

    accuracy                           0.85     20632
   macro avg       0.84      0.86      0.85     20632
weighted avg       0.85      0.85      0.85     20632

CONFUSSION MATRIX
[[2839  373   19]
 [ 699 8480 1163]
 [  34  805 6220]]
```
#### Performance After 20% Augmentation
```python
REPORT
              precision    recall  f1-score   support

         NEG       0.73      0.76      0.75      3231
         NEU       0.79      0.83      0.81     10342
         POS       0.84      0.76      0.80      7059

    accuracy                           0.80     20632
   macro avg       0.79      0.78      0.79     20632
weighted avg       0.80      0.80      0.80     20632

CONFUSSION MATRIX
[[2465  709   57]
 [ 798 8560  984]
 [  93 1571 5395]]
```
#### Performance After 50% Augmentation
```python
REPORT
              precision    recall  f1-score   support

         NEG       0.66      0.61      0.63      3231
         NEU       0.69      0.85      0.76     10342
         POS       0.83      0.57      0.68      7059

    accuracy                           0.72     20632
   macro avg       0.73      0.68      0.69     20632
weighted avg       0.73      0.72      0.71     20632

CONFUSSION MATRIX
[[1961 1209   61]
 [ 789 8801  752]
 [ 200 2834 4025]]
```

### Optimized Augmentation on `siebert/sentiment-roberta-large-english` with `rotten_tomatoes` dataset (CLI)

#### Claimed Performance 

#### Performance under attack 
```c
Attack(
  (search_method): GreedyWordSwapWIR(
    (wir_method):  unk
  )
  (goal_function):  UntargetedClassification
  (transformation):  WordSwapEmbedding(
    (max_candidates):  15
    (embedding):  WordEmbedding
  )
  (constraints): 
    (0): RepeatModification
    (1): StopwordModification
  (is_black_box):  True
) 
```

```c 
Results
+-------------------------------+--------+
| Attack Results                |        |
+-------------------------------+--------+
| Number of successful attacks: | 15     |
| Number of failed attacks:     | 0      |
| Number of skipped attacks:    | 0      |
| Original accuracy:            | 100.0% |
| Accuracy under attack:        | 0.0%   |
| Attack success rate:          | 100.0% |
| Average perturbed word %:     | 18.6%  |
| Average num. words per input: | 21.33  |
| Avg num queries:              | 69.27  |
+-------------------------------+--------+
```

### Optimized Augmentation on `siebert/sentiment-roberta-large-english` with `IMDB` dataset with wrapper

#### Claimed Performance 

#### Performance under attack 

```c
Attack(
  (search_method): GreedySearch
  (goal_function):  UntargetedClassification
  (transformation):  WordSwapEmbedding(
    (max_candidates):  15
    (embedding):  WordEmbedding
  )
  (constraints): 
    (0): RepeatModification
    (1): StopwordModification
  (is_black_box):  True
) 
```

```c 
+-------------------------------+---------+
| Attack Results                |         |
+-------------------------------+---------+
| Number of successful attacks: | 10      |
| Number of failed attacks:     | 0       |
| Number of skipped attacks:    | 0       |
| Original accuracy:            | 100.0%  |
| Accuracy under attack:        | 0.0%    |
| Attack success rate:          | 100.0%  |
| Average perturbed word %:     | 5.19%   |
| Average num. words per input: | 200.5   |
| Avg num queries:              | 16856.5 |
+-------------------------------+---------+
```



## DETAILED OUTPUT 
### EDA ON rotten tomatoes +sentiment-roberta-large-english
```text
--------------------------------------------- Result 1 ---------------------------------------------
[[Positive (100%)]] --> [[Negative (96%)]]

it's an ambitious film , and as with all ambitious [[films]] , it has some problems . but on the [[whole]] , you're [[gonna]] [[like]] this [[movie]] .

it's an ambitious film , and as with all ambitious [[cine]] , it has some problems . but on the [[generals]] , you're [[will]] [[genera]] this [[cinematography]] .
--------------------------------------------- Result 2 ---------------------------------------------
[[Negative (100%)]] --> [[Positive (99%)]]

cacoyannis' vision is far [[less]] mature , interpreting the play as a call for pity and sympathy for anachronistic phantasms haunting the imagined glory of their own pasts .

cacoyannis' vision is far [[smaller]] mature , interpreting the play as a call for pity and sympathy for anachronistic phantasms haunting the imagined glory of their own pasts .
--------------------------------------------- Result 3 ---------------------------------------------
[[Positive (100%)]] --> [[Negative (98%)]]

yeah , these flicks are just that damn [[good]] . isn't it great ?

yeah , these flicks are just that damn [[alright]] . isn't it great ?
--------------------------------------------- Result 4 ---------------------------------------------
[[Positive (100%)]] --> [[Negative (100%)]]

this movie [[got]] me [[grinning]] . there's a part of us that cannot help be entertained by the sight of someone getting away with something .

this movie [[was]] me [[smirking]] . there's a part of us that cannot help be entertained by the sight of someone getting away with something .
--------------------------------------------- Result 5 ---------------------------------------------
[[Negative (100%)]] --> [[Positive (99%)]]

what the [[director]] can&#8217 ; t do is [[make]] [[either]] of val [[kilmer]]&#8217 ; s two [[personas]] [[interesting]] or worth caring about .

what the [[headmistress]] can&#8217 ; t do is [[afford]] [[without]] of val [[harrelson]]&#8217 ; s two [[personally]] [[riveting]] or worth caring about .
--------------------------------------------- Result 6 ---------------------------------------------
[[Positive (100%)]] --> [[Negative (100%)]]

the passions aroused by the discord between old and new cultures are [[set]] against the strange , stark beauty of the mideast desert , so [[lovingly]] and perceptively [[filmed]] that you can almost taste the desiccated air .

the passions aroused by the discord between old and new cultures are [[gaming]] against the strange , stark beauty of the mideast desert , so [[lavishly]] and perceptively [[murdered]] that you can almost taste the desiccated air .
--------------------------------------------- Result 7 ---------------------------------------------
[[Negative (100%)]] --> [[Positive (99%)]]

trouble [[every]] day is a success in some sense , but it's [[hard]] to like a [[film]] so cold and [[dead]] .

trouble [[eveything]] day is a success in some sense , but it's [[challenging]] to like a [[cinematography]] so cold and [[die]] .
--------------------------------------------- Result 8 ---------------------------------------------
[[Negative (100%)]] --> [[Positive (97%)]]

there's no [[getting]] around the [[fact]] that this is revenge of the [[nerds]] [[revisited]] -- again .

there's no [[obtain]] around the [[effected]] that this is revenge of the [[geeks]] [[scrutinize]] -- again .
--------------------------------------------- Result 9 ---------------------------------------------
[[Negative (100%)]] --> [[Positive (99%)]]

oh [[come]] on . [[like]] you couldn't [[smell]] this [[turkey]] rotting from [[miles]] away .

oh [[be]] on . [[enjoy]] you couldn't [[perfume]] this [[armenians]] rotting from [[milla]] away .
--------------------------------------------- Result 10 ---------------------------------------------
[[Negative (100%)]] --> [[Positive (98%)]]

shreve's [[graceful]] [[dual]] [[narrative]] gets [[clunky]] on the screen , and we keep [[getting]] [[torn]] away from the compelling [[historical]] tale to a less-compelling [[soap]] [[opera]] .

shreve's [[graciously]] [[doubled]] [[tale]] gets [[fiddly]] on the screen , and we keep [[gain]] [[divided]] away from the compelling [[landmark]] tale to a less-compelling [[soaps]] [[theaters]] .
--------------------------------------------- Result 11 ---------------------------------------------
[[Positive (100%)]] --> [[Negative (100%)]]

the film is [[darkly]] [[funny]] in its observation of just how much more grueling and time-consuming the illusion of work is than actual work .

the film is [[comically]] [[odd]] in its observation of just how much more grueling and time-consuming the illusion of work is than actual work .
--------------------------------------------- Result 12 ---------------------------------------------
[[Negative (100%)]] --> [[Positive (99%)]]

only about as [[sexy]] and dangerous as an actress in a role that [[reminds]] at [[every]] [[turn]] of elizabeth berkley's [[flopping]] dolphin-gasm .

only about as [[scorching]] and dangerous as an actress in a role that [[callbacks]] at [[tous]] [[transforms]] of elizabeth berkley's [[wobbling]] dolphin-gasm .
--------------------------------------------- Result 13 ---------------------------------------------
[[Negative (100%)]] --> [[Positive (99%)]]

as [[exciting]] as all this exoticism might [[sound]] to the [[typical]] [[pax]] [[viewer]] , the [[rest]] of us will be [[lulled]] into a coma .

as [[breathtaking]] as all this exoticism might [[audible]] to the [[hallmarks]] [[romulus]] [[aimer]] , the [[resting]] of us will be [[mesmerized]] into a coma .
--------------------------------------------- Result 14 ---------------------------------------------
[[Negative (100%)]] --> [[Positive (99%)]]

a [[humorless]] journey into a philosophical void .

a [[supercilious]] journey into a philosophical void .
--------------------------------------------- Result 15 ---------------------------------------------
[[Negative (100%)]] --> [[Positive (99%)]]

i've heard that the fans of the first men in black have come away hating the second one . i wonder why . they [[felt]] like the same movie to me .

i've heard that the fans of the first men in black have come away hating the second one . i wonder why . they [[believed]] like the same movie to me .

-------------------------------------
Number of successful attacks: 15
Number of failed attacks: 0
Number of skipped attacks: 0
Original accuracy: 100.0%
Accuracy under attack: 0.0%
Attack success rate: 100.0%
Average perturbed word %: 18.6%
Average num. words per input: 21.33
Avg num queries: 69.27
-------------------------------------
```

### EDA ON IMDB+sentiment-roberta-large-english
```text
textattack: Logging to CSV at path log_28April.csv
textattack: Running 1 worker(s) on 1 GPU(s).
textattack: Worklist size: 10
textattack: Worklist candidate size: 39990
  0%|          | 0/10 [00:00<?, ?it/s]
Attack(
  (search_method): GreedySearch
  (goal_function):  UntargetedClassification
  (transformation):  WordSwapEmbedding(
    (max_candidates):  15
    (embedding):  WordEmbedding
  )
  (constraints): 
    (0): RepeatModification
    (1): StopwordModification
  (is_black_box):  True
) 

[Succeeded / Failed / Skipped / Total] 1 / 0 / 0 / 1:  10%|█         | 1/10 [06:49<1:01:27, 409.72s/it]
--------------------------------------------- Result 1 ---------------------------------------------
[[0 (100%)]] --> [[1 (98%)]]

I grew up (b. 1965) watching and loving the Thunderbirds. All my mates at school watched. We played "Thunderbirds" before school, during lunch and after school. We all wanted to be Virgil or Scott. No one wanted to be Alan. Counting down from 5 became an art form. I took my children to see the [[movie]] hoping they [[would]] get a glimpse of what I loved as a child. How [[bitterly]] [[disappointing]]. The only [[high]] point was the snappy theme tune. Not that it could compare with the original score of the Thunderbirds. Thankfully early Saturday mornings one television channel still plays reruns of the series Gerry Anderson and his wife created. Jonatha Frakes should hand in his [[directors]] chair, his version was completely [[hopeless]]. A [[waste]] of film. Utter [[rubbish]]. A CGI remake may be acceptable but replacing marionettes with Homo sapiens subsp. sapiens was a huge error of judgment.

I grew up (b. 1965) watching and loving the Thunderbirds. All my mates at school watched. We played "Thunderbirds" before school, during lunch and after school. We all wanted to be Virgil or Scott. No one wanted to be Alan. Counting down from 5 became an art form. I took my children to see the [[theater]] hoping they [[cannot]] get a glimpse of what I loved as a child. How [[keenly]] [[sad]]. The only [[altos]] point was the snappy theme tune. Not that it could compare with the original score of the Thunderbirds. Thankfully early Saturday mornings one television channel still plays reruns of the series Gerry Anderson and his wife created. Jonatha Frakes should hand in his [[headmaster]] chair, his version was completely [[aghast]]. A [[offal]] of film. Utter [[litter]]. A CGI remake may be acceptable but replacing marionettes with Homo sapiens subsp. sapiens was a huge error of judgment.
[Succeeded / Failed / Skipped / Total] 2 / 0 / 0 / 2:  20%|██        | 2/10 [23:39<1:34:37, 709.66s/it]
--------------------------------------------- Result 2 ---------------------------------------------
[[0 (100%)]] --> [[1 (97%)]]

When I put this movie in my DVD player, and sat down with a coke and some chips, I had some [[expectations]]. I was hoping that this movie would contain some of the strong-points of the first movie: Awsome animation, good flowing story, excellent voice cast, funny comedy and a kick-ass soundtrack. [[But]], to my [[disappointment]], not any of this is to be found in [[Atlantis]]: Milo's Return. Had I read some reviews first, I might not have been so let down. The following paragraph will be directed to those who have [[seen]] the first movie, and who enjoyed it primarily for the points mentioned.<br /><br />When the first scene appears, your in for a [[shock]] if you just picked Atlantis: Milo's Return from the display-case at your local videoshop (or whatever), and had the expectations I had. The music [[feels]] as a bad imitation of the first movie, and the voice cast has been replaced by a not so fitting one. (With the exception of a few characters, like the voice of Sweet). The actual drawings isnt that bad, but the animation in particular is a [[sad]] sight. The storyline is also [[pretty]] [[weak]], as its more like three episodes of Schooby-Doo than the single adventurous story we got the last time. But dont misunderstand, it's not very [[good]] Schooby-Doo episodes. I didnt [[laugh]] a single time, although I might have sniggered once or twice.<br /><br />To the audience who haven't seen the first movie, or don't especially care for a similar sequel, here is a fast review of this movie as a stand-alone product: If you liked schooby-doo, you might like this movie. If you didn't, you could still enjoy this movie if you have nothing else to do. And I suspect it might be a good kids movie, but I wouldn't know. It might have been better if Milo's Return had been a three-episode series on a cartoon channel, or on breakfast TV.

When I put this movie in my DVD player, and sat down with a coke and some chips, I had some [[anticipation]]. I was hoping that this movie would contain some of the strong-points of the first movie: Awsome animation, good flowing story, excellent voice cast, funny comedy and a kick-ass soundtrack. [[Although]], to my [[bummer]], not any of this is to be found in [[Galactica]]: Milo's Return. Had I read some reviews first, I might not have been so let down. The following paragraph will be directed to those who have [[noticed]] the first movie, and who enjoyed it primarily for the points mentioned.<br /><br />When the first scene appears, your in for a [[amazement]] if you just picked Atlantis: Milo's Return from the display-case at your local videoshop (or whatever), and had the expectations I had. The music [[thought]] as a bad imitation of the first movie, and the voice cast has been replaced by a not so fitting one. (With the exception of a few characters, like the voice of Sweet). The actual drawings isnt that bad, but the animation in particular is a [[triste]] sight. The storyline is also [[lovely]] [[vulnerable]], as its more like three episodes of Schooby-Doo than the single adventurous story we got the last time. But dont misunderstand, it's not very [[opportune]] Schooby-Doo episodes. I didnt [[kidding]] a single time, although I might have sniggered once or twice.<br /><br />To the audience who haven't seen the first movie, or don't especially care for a similar sequel, here is a fast review of this movie as a stand-alone product: If you liked schooby-doo, you might like this movie. If you didn't, you could still enjoy this movie if you have nothing else to do. And I suspect it might be a good kids movie, but I wouldn't know. It might have been better if Milo's Return had been a three-episode series on a cartoon channel, or on breakfast TV.
[Succeeded / Failed / Skipped / Total] 3 / 0 / 0 / 3:  30%|███       | 3/10 [27:30<1:04:11, 550.28s/it]
--------------------------------------------- Result 3 ---------------------------------------------
[[0 (100%)]] --> [[1 (99%)]]

Why do people who do not know what a particular time in the past was like feel the need to try to define that time for others? Replace Woodstock with the Civil War and the Apollo moon-landing with the Titanic sinking and you've got as realistic a flick as this formulaic soap [[opera]] populated [[entirely]] by low-life trash. Is this what kids who were too young to be allowed to go to Woodstock and who failed grade school composition do? "I'll show those old meanies, I'll put out my own movie and prove that you don't have to know nuttin about your topic to still make money!" Yeah, we already know that. The [[one]] thing watching this film did for me was to give me a little [[insight]] into underclass thinking. The next time I see a slut in a bar who looks like Diane Lane, I'm [[running]] the other way. It's child abuse to let parents that worthless raise kids. It's audience abuse to simply stick Woodstock and the moonlanding into a flick as if that ipso facto means the film portrays 1969.

Why do people who do not know what a particular time in the past was like feel the need to try to define that time for others? Replace Woodstock with the Civil War and the Apollo moon-landing with the Titanic sinking and you've got as realistic a flick as this formulaic soap [[premiered]] populated [[perfectly]] by low-life trash. Is this what kids who were too young to be allowed to go to Woodstock and who failed grade school composition do? "I'll show those old meanies, I'll put out my own movie and prove that you don't have to know nuttin about your topic to still make money!" Yeah, we already know that. The [[uno]] thing watching this film did for me was to give me a little [[outlook]] into underclass thinking. The next time I see a slut in a bar who looks like Diane Lane, I'm [[implement]] the other way. It's child abuse to let parents that worthless raise kids. It's audience abuse to simply stick Woodstock and the moonlanding into a flick as if that ipso facto means the film portrays 1969.
[Succeeded / Failed / Skipped / Total] 4 / 0 / 0 / 4:  40%|████      | 4/10 [29:48<44:42, 447.06s/it]  
--------------------------------------------- Result 4 ---------------------------------------------
[[0 (100%)]] --> [[1 (99%)]]

Even though I have great interest in Biblical movies, I was [[bored]] to death every minute of the movie. [[Everything]] is [[bad]]. [[The]] movie is too long, the acting is most of the time a Joke and the script is [[horrible]]. [[I]] did not get the point in mixing the story about Abraham and Noah together. So if you value your time and sanity stay [[away]] from this [[horror]].

Even though I have great interest in Biblical movies, I was [[piercing]] to death every minute of the movie. [[Each]] is [[wicked]]. [[Of]] movie is too long, the acting is most of the time a Joke and the script is [[grisly]]. [[ya]] did not get the point in mixing the story about Abraham and Noah together. So if you value your time and sanity stay [[beyond]] from this [[fear]].
[Succeeded / Failed / Skipped / Total] 5 / 0 / 0 / 5:  50%|█████     | 5/10 [39:41<39:41, 476.36s/it]
--------------------------------------------- Result 5 ---------------------------------------------
[[1 (100%)]] --> [[0 (99%)]]

Im a die hard Dads Army fan and nothing will ever change that. I got all the tapes, DVD's and audiobooks and [[every]] time i watch/listen to them its brand new. <br /><br />[[The]] film. The film is a re run of certain [[episodes]], Man and the hour, Enemy within the gates, Battle School and numerous others with a different [[edge]]. Introduction of a new General instead of Captain Square was a brilliant move - especially when he wouldn't cash the cheque (something that is rarely done now).<br /><br />It [[follows]] through the early years of getting equipment and uniforms, starting up and training. All in all, its a [[great]] film for a boring Sunday afternoon. <br /><br />Two [[draw]] backs. One is the [[Germans]] bogus dodgy accents (come one, Germans cant pronounced the letter "W" like us) and Two The casting of Liz Frazer [[instead]] of the familiar Janet Davis. I like Liz in other films like the carry ons but she doesn't [[carry]] it correctly in this and Janet Davis would have been the [[better]] choice.

Im a die hard Dads Army fan and nothing will ever change that. I got all the tapes, DVD's and audiobooks and [[whole]] time i watch/listen to them its brand new. <br /><br />[[Al]] film. The film is a re run of certain [[spasms]], Man and the hour, Enemy within the gates, Battle School and numerous others with a different [[rand]]. Introduction of a new General instead of Captain Square was a brilliant move - especially when he wouldn't cash the cheque (something that is rarely done now).<br /><br />It [[remained]] through the early years of getting equipment and uniforms, starting up and training. All in all, its a [[whopping]] film for a boring Sunday afternoon. <br /><br />Two [[attracting]] backs. One is the [[Tak]] bogus dodgy accents (come one, Germans cant pronounced the letter "W" like us) and Two The casting of Liz Frazer [[alternately]] of the familiar Janet Davis. I like Liz in other films like the carry ons but she doesn't [[take]] it correctly in this and Janet Davis would have been the [[improving]] choice.
[Succeeded / Failed / Skipped / Total] 6 / 0 / 0 / 6:  60%|██████    | 6/10 [41:07<27:25, 411.27s/it]
--------------------------------------------- Result 6 ---------------------------------------------
[[0 (100%)]] --> [[1 (100%)]]

A [[terrible]] movie as everyone has said. What made me laugh was the cameo appearance by Scott McNealy, giving an award to one of the murdered programmers in front of a wall of SUN logos. McNealy is the CEO of SUN Microsystem, a company that practically defines itself by its hatred of Microsoft. They have been instrumental in filing antitrust complaints against Microsoft. So, were they silly enough to think this [[bad]] movie would add fuel to that fire?<br /><br />There's no public record I see of SUN's involvement, but clearly the makers of this movie know Scott McNealy. An [[interesting]] mystery.

A [[grisly]] movie as everyone has said. What made me laugh was the cameo appearance by Scott McNealy, giving an award to one of the murdered programmers in front of a wall of SUN logos. McNealy is the CEO of SUN Microsystem, a company that practically defines itself by its hatred of Microsoft. They have been instrumental in filing antitrust complaints against Microsoft. So, were they silly enough to think this [[naughty]] movie would add fuel to that fire?<br /><br />There's no public record I see of SUN's involvement, but clearly the makers of this movie know Scott McNealy. An [[riveting]] mystery.
[Succeeded / Failed / Skipped / Total] 7 / 0 / 0 / 7:  70%|███████   | 7/10 [44:14<18:57, 379.19s/it]
--------------------------------------------- Result 7 ---------------------------------------------
[[1 (100%)]] --> [[0 (100%)]]

Finally watched this [[shocking]] movie last night, and what a disturbing mindf**ker it is, and unbelievably bloody and some unforgettable scenes, and a total assault on the senses. [[Looks]] [[like]] a movie from the minds of Lynch (specifically ERASERHEAD), Buttgereit, and even a little of "Begotten". What this guy does to his pregnant sister is beyond belief, but then again, did it really happen or is it his brain's left and right sides doing battle. That's the main theme of this piece of art, to draw a fine line between fantasy and reality, and what would happen if the right side of the brain that dreams and fantasizes overtakes the reasoning and logical left side. And the music in this movie is unbelievable, a kind of electronic score that is absolutely perfect. Even though this movie is totally shocking and pretty disgusting in some of the most extreme scenes (including hard core sex) you will ever see in any movie, I viewed it as a work of art, and loved it. And that music still amazes me, I have to try and find the soundtrack if is available. Watching "Subconscious Cruelty" is a real event, and not something the viewer will easily forget. And a note to gorehounds, this is a must-have.<br /><br />Warning... Be careful buying this movie, because some prints have fogging on the graphic sex scenes and extreme gore, especially the copies from the Japanese release.

Finally watched this [[terrible]] movie last night, and what a disturbing mindf**ker it is, and unbelievably bloody and some unforgettable scenes, and a total assault on the senses. [[Expecting]] [[enjoy]] a movie from the minds of Lynch (specifically ERASERHEAD), Buttgereit, and even a little of "Begotten". What this guy does to his pregnant sister is beyond belief, but then again, did it really happen or is it his brain's left and right sides doing battle. That's the main theme of this piece of art, to draw a fine line between fantasy and reality, and what would happen if the right side of the brain that dreams and fantasizes overtakes the reasoning and logical left side. And the music in this movie is unbelievable, a kind of electronic score that is absolutely perfect. Even though this movie is totally shocking and pretty disgusting in some of the most extreme scenes (including hard core sex) you will ever see in any movie, I viewed it as a work of art, and loved it. And that music still amazes me, I have to try and find the soundtrack if is available. Watching "Subconscious Cruelty" is a real event, and not something the viewer will easily forget. And a note to gorehounds, this is a must-have.<br /><br />Warning... Be careful buying this movie, because some prints have fogging on the graphic sex scenes and extreme gore, especially the copies from the Japanese release.
[Succeeded / Failed / Skipped / Total] 8 / 0 / 0 / 8:  80%|████████  | 8/10 [1:16:50<19:12, 576.30s/it]
--------------------------------------------- Result 8 ---------------------------------------------
[[0 (100%)]] --> [[1 (99%)]]

I caught this film on AZN on cable. It [[sounded]] like it would be a [[good]] film, a Japanese "Green Card". I can't say I've ever disliked an Asian film, quite the contrary. Some of the most incredible horror films of all time are Japanese and Korean, and I am a HUGE fan of John Woo's Hong Kong films. I an not adverse to a light hearted films, like Tampopo or Chung King Express (two of my favourites), so I [[thought]] I [[would]] [[like]] this. [[Well]], I [[would]] [[rather]] [[slit]] my wrists and drink my own blood than watch this [[laborious]], [[badly]] [[acted]] film ever again.<br /><br />[[I]] think the director Steven Okazaki [[must]] have [[spiked]] the [[water]] with Quaalude, because no one in this film had a personality. And when any of the characters DID try to act, as opposed to mumbling a line or two, their performance came [[across]] as forced and incredibly fake. I honestly did not think that anyone had ever acted before...the ONLY person who sounded genuine was Brenda Aoki.. I find it amazing that this is promoted as a [[comedy]], because I didn't laugh once. [[Even]] MORE surprising is that CBS morning news called this "a refreshing breath of comedy". It was [[neither]] refreshing, nor a breath of comedy. And the ending was very [[predictable]], the previous reviewer must be an idiot to think such [[things]].<br /><br />[[AVOID]] this film unless you want to [[see]] a boring predictable plot line and wooden acting. I actually think that "Spike of Bensonhurst" is a better acted film than this...and I [[walked]] out half way through that film!

I caught this film on AZN on cable. It [[called]] like it would be a [[guten]] film, a Japanese "Green Card". I can't say I've ever disliked an Asian film, quite the contrary. Some of the most incredible horror films of all time are Japanese and Korean, and I am a HUGE fan of John Woo's Hong Kong films. I an not adverse to a light hearted films, like Tampopo or Chung King Express (two of my favourites), so I [[believe]] I [[was]] [[loved]] this. [[Good]], I [[cannot]] [[instead]] [[cracking]] my wrists and drink my own blood than watch this [[tricky]], [[seriously]] [[worked]] film ever again.<br /><br />[[did]] think the director Steven Okazaki [[required]] have [[immune]] the [[shui]] with Quaalude, because no one in this film had a personality. And when any of the characters DID try to act, as opposed to mumbling a line or two, their performance came [[under]] as forced and incredibly fake. I honestly did not think that anyone had ever acted before...the ONLY person who sounded genuine was Brenda Aoki.. I find it amazing that this is promoted as a [[hilarious]], because I didn't laugh once. [[Nonetheless]] MORE surprising is that CBS morning news called this "a refreshing breath of comedy". It was [[either]] refreshing, nor a breath of comedy. And the ending was very [[forecasting]], the previous reviewer must be an idiot to think such [[issues]].<br /><br />[[PREVENTING]] this film unless you want to [[look]] a boring predictable plot line and wooden acting. I actually think that "Spike of Bensonhurst" is a better acted film than this...and I [[acted]] out half way through that film!
[Succeeded / Failed / Skipped / Total] 9 / 0 / 0 / 9:  90%|█████████ | 9/10 [1:26:38<09:37, 577.64s/it]
--------------------------------------------- Result 9 ---------------------------------------------
[[1 (100%)]] --> [[0 (99%)]]

[[It]] [[may]] be the [[remake]] of 1987 Autumn's Tale after eleven [[years]], as the director Mabel Cheung claimed. Mabel employs rock music as the medium in this movie to express her personal attitude to life, in which love, desire and the consequential frustration play significantly crucial roles. Rock music may not be the best vehicle to convey the profound sentiment, and yet it is not too inappropriate to utilize it as the life of underground rock musicians is bitterly more intense than an ordinary one. The director focuses on the depiction of subtle affection and ultimate vanity of life [[rather]] than mere rock music. The love between father and son, lovers, and friends is [[delicately]] and [[touchingly]] delivered through the fine performance. [[Mabel]] does not attempt to beautify rock musicians as artists at all, instead, she tries to reproduce a [[true]] [[life]] on screen, [[making]] huge efforts of years' working on this project and gathering information in Beijing underground pubs.<br /><br />Daniel has given probably the best performance in all his movies made so far. His innate dispiritedness and reticence fit the blue mood of the film perfectly.

[[That]] [[likely]] be the [[redone]] of 1987 Autumn's Tale after eleven [[ages]], as the director Mabel Cheung claimed. Mabel employs rock music as the medium in this movie to express her personal attitude to life, in which love, desire and the consequential frustration play significantly crucial roles. Rock music may not be the best vehicle to convey the profound sentiment, and yet it is not too inappropriate to utilize it as the life of underground rock musicians is bitterly more intense than an ordinary one. The director focuses on the depiction of subtle affection and ultimate vanity of life [[slightly]] than mere rock music. The love between father and son, lovers, and friends is [[mildly]] and [[mawkish]] delivered through the fine performance. [[Gladys]] does not attempt to beautify rock musicians as artists at all, instead, she tries to reproduce a [[trusty]] [[vie]] on screen, [[fabricating]] huge efforts of years' working on this project and gathering information in Beijing underground pubs.<br /><br />Daniel has given probably the best performance in all his movies made so far. His innate dispiritedness and reticence fit the blue mood of the film perfectly.
[Succeeded / Failed / Skipped / Total] 10 / 0 / 0 / 10: 100%|██████████| 10/10 [1:36:30<00:00, 579.06s/it]
--------------------------------------------- Result 10 ---------------------------------------------
[[1 (100%)]] --> [[0 (99%)]]

My Super Ex Girlfriend turned out to be a pleasant surprise for me, I was really expecting a horrible movie that would probably be stupid and predictable, and you know what? It was! [[But]] this movie did have so many wonderful laughs and a fun plot that anyone could get a kick out of. I know that this was a very cheesy movie, but Uma and Anna were just so [[cool]] and Steve was such a great addition along with a great cast that looked like they had so much fun and that's what [[made]] the movie [[really]] [[work]].<br /><br />Jenny Johnson(scary, that's my best friend's actual name) is not your typical average librarian looking woman, when Matt, your average male, asks her out, he's in for more than he expected, he's asked G-Girl out on a date, the super hero of the world! But when he finds out what a jealous and crazy girl she really is and decides that it may be a good idea that they spend some time apart, but Jenny won't have it since he's fallen for another girl, Hannah, and she will make his life a living hell, I mean, let's face it, he couldn't have chosen a better girl to break up with.<br /><br />The effect were corny, but you seriously move past them quickly, the story and cast made the story really [[work]] and I [[loved]] Uma in this movie, it was such a step up from Prime. [[My]] Super Ex Girlfriend is a fun movie that you shouldn't really take seriously, it's just a cute romantic comedy that I think if I could get a laugh out of it, anyone could.<br /><br />7/10

My Super Ex Girlfriend turned out to be a pleasant surprise for me, I was really expecting a horrible movie that would probably be stupid and predictable, and you know what? It was! [[If]] this movie did have so many wonderful laughs and a fun plot that anyone could get a kick out of. I know that this was a very cheesy movie, but Uma and Anna were just so [[cold]] and Steve was such a great addition along with a great cast that looked like they had so much fun and that's what [[yielded]] the movie [[absolutely]] [[labor]].<br /><br />Jenny Johnson(scary, that's my best friend's actual name) is not your typical average librarian looking woman, when Matt, your average male, asks her out, he's in for more than he expected, he's asked G-Girl out on a date, the super hero of the world! But when he finds out what a jealous and crazy girl she really is and decides that it may be a good idea that they spend some time apart, but Jenny won't have it since he's fallen for another girl, Hannah, and she will make his life a living hell, I mean, let's face it, he couldn't have chosen a better girl to break up with.<br /><br />The effect were corny, but you seriously move past them quickly, the story and cast made the story really [[labor]] and I [[amour]] Uma in this movie, it was such a step up from Prime. [[Mona]] Super Ex Girlfriend is a fun movie that you shouldn't really take seriously, it's just a cute romantic comedy that I think if I could get a laugh out of it, anyone could.<br /><br />7/10
[Succeeded / Failed / Skipped / Total] 10 / 0 / 0 / 10: 100%|██████████| 10/10 [1:36:31<00:00, 579.12s/it]

+-------------------------------+---------+
| Attack Results                |         |
+-------------------------------+---------+
| Number of successful attacks: | 10      |
| Number of failed attacks:     | 0       |
| Number of skipped attacks:    | 0       |
| Original accuracy:            | 100.0%  |
| Accuracy under attack:        | 0.0%    |
| Attack success rate:          | 100.0%  |
| Average perturbed word %:     | 5.19%   |
| Average num. words per input: | 200.5   |
| Avg num queries:              | 16856.5 |
+-------------------------------+---------+
```