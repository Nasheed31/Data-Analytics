#!/usr/bin/env python
# coding: utf-8

# In[4]:


#6th url
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text = 'IntroductionWhere is this disruptive technology taking us? Take it or leave it, disruptive technology always creates new jobs much more than depleted jobs. You might notice certain jobs disappearing but those jobs are the jobs that transform humans to robots, to machines, and the technology is creating machines to replace them.\xa0 Technology creates the data analysis tools to manipulate and create custom scenarios using artificial intelligence (AI), Big Data and Machine Learning (ML) algorithms to predict and drive consumer behavior. Data Analytics tools, such as Google Analytics , and others are available today for free, and, if used correctly, can help organizations save millions, maybe billions of dollars of sales and marketing.How machine will replace humans?Before I go on, I think it’s best to level set on what constitutes\xa0machines.\xa0In the context of this article , machines describe computers and computerized equipment, like robots, that have been programmed to learn, sometimes like humans. Occasionally we call this Artificial Intelligence (AI), other times we call this machine learning, and still other times we call this robotics. And yes, these are technically different things. These bots are more efficient than humans in some specific domains and are growing smarter with each passing day. They can do some really tough tasks which are considered difficult for any human being. \xa0But, within the broad discussion related to the future of work, these are totally interrelated. Factory floors deploy robots that are increasingly driven by machine learning algorithms such that they can adjust to people working alongside them. A machine can work efficiently only it has abundant data and information about the work which is being imparted daily to them. But with every forward step & advancement in technology, a threat is proliferating, a threat of being replaced on our work front. Every passing day is sealing some jobs for humans all over the globe. Similarly, AI is being used to turn hand-drawn sketches (done by humans) into digital source code.Role of Machines in Companies and its future:Companies are clearly developing their AI and robotics expertise with the idea that through these technological innovations they’ll be able tocut costsincrease efficienciesoffer new value propositionsexecute new business models or all of the above.Of course, it’s not just machines and creatives working together either. In another example, Amazon has employed more than 100,000 robots in its warehouses to efficiently move things around while it has increased its warehouse workforce by more than 80,000. Humans, in Amazon’s case, do the picking and packing of goods while robots move orders around the giant warehouses, essentially cutting “down on the walking required of workers, making Amazon pickers more efficient and less tired.” Plus, the robots “allow Amazon to pack shelves together like cars in rush-hour traffic because they no longer need aisle space for humans. The greater density of shelf space means more inventory under one roof, which means better selection for customers.”Why Machines can replace humans?During the next few decades (or maybe sooner), the notion of work and whether it is handled by a human or a virtual being will hinge on predictability. As they are starting to do today, machines will manage the routine while humans take on the unpredictable – tasks that require creativity, problem-solving, and flexibility. In this context, robotics should be seen not only as a means to improve operations efficiency but also to improve the quality of life for workers.Although it is obvious that human factors involved in a work activity impact job automation, it is also true that highly repetitive tasks—and even mechanical ones—are ideal for robots. Besides greater efficiency and speed, automation leads to a lower risk of accidents, greater control and autonomy, and above all, fewer costs for organizations.Are Machines more diligent than humans?Although artificial intelligence and machine learning make us believe that robots are endowed with superior intelligence, in fact they don’t yet have the ability to learn from experience and to respond to unknowns. So as things stand, however much processing speed and automatic learning a robot has, it doesn’t beat factors innate to the human brain. Humans are still a very essential part of the process. Think about delivering services to a client. Most customer challenges are routine, but humans play a very important role in addressing new issues, solving them the first time they appear, and then consolidating the process into the system.Machines vs. Humans – Which is premium?While machines and humans are placed in proximity, \xa0robots can be expensive, but this doesn’t apply to all types, especially those based on Robotic Process Automation (RPA), where the development process incorporates algorithms that significantly reduce costs.Moreover, think of how domestic robots—be it a vacuum cleaner, a lawn mower or a pool cleaner—are increasingly part of our daily lives. This level of consumption that robotics has attained makes it affordable to automate tasks in modern homes to obtain greater control, security and comfort.Man and Machines togetherThe division between humans and machines has been clear – I’m here, the machine is there – but that boundary is getting fuzzier. Smart prosthetics fuse seamlessly with our bodies, making up for lost limbs or providing additional strength, stability, or resilience, as seen in exoskeletons donned by assembly line workers.We use our smartphones symbiotically, but what if they were integrated directly into our bodies? Think a smartphone in the form of a contact lens capable of transparently delivering augmented reality images straight to the brain. Think it sounds like science fiction? Think again. The first prototypes have already been built.Soon, brain-computer interfaces could become seamless as well, creating a new synergistic relationship between the cloud and us. At that point, the question of who knows what would be moot; you ask me a question and I know the answer. Sometimes that answer will be stored in my own neural circuitry, but most of the time it would come from the connection of my neurons to the web. Our brain’s decision process is influenced by the way it has been “educated” by the cultural context. These external factors are influencing our decision processes to the point that in certain situations, we can legitimately claim that influence has been so strong that our brains can’t be held accountable for the choices made. The point I’m trying to make is that we humans are in symbiosis with our cultural environment and the tools – both physical and conceptual – that we have been taught to use. My guess is that the transformation will be subtle.ConclusionPractically speaking, robots growing to the point that they take over the world and then start creating smarter, better robots are impractical and should not even be a concern. None of this is expected in the near future, not by a long shot. If you’ve been to an ATM, waited for a PC to boot up after a catastrophic failure, or had a game crash on your X box just when you were about to reach a checkpoint, you understand that we are not in a world where machines do everything perfectly right. Before they can take over all of our jobs, they need to be able to do theirs’ flawlessly; until then, we can depend on humans to mess up our lives.\xa0This isn’t a win-or-lose situation. We’re going to wind up as a partner to our smarter machines, and that partnership will be fostered by our augmentation through technology. Machines will play an essential role in this augmentation and, as with any successful technology, they will fall below our level of perception. In the end, the revolution will be silent and invisible.Blackcoffer Insights 29: Swapna. G, Nivashiniya. R, Sri Manakula Vinayagar Engineering College(SMVEC), Puducherry'
# Sentiment analysis
sia = SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']

# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[5]:


#7
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text = 'In future or in upcoming years humans and machines are going to work together in every field of work. In upcoming days machines will be the need for every human being. Machines [AI technology] will do the work which humans are incapable of doing. Machines will partner and co-operate with humans.According to the professor at the university of Washington, he explained that, as a result of AI, there will be more demand for existing jobs and new jobs will be created that are unimaginable today. Human workers and machines will work together flawlessly, complementing each other. Machines will learn to carry out easier tasks such as following processes or crunching data. They will also help the humans while difficult. Machines or AI will create a great job opportunities for humans in future. John Kelly ll, executive vice president of IBM once said that “Man and Machines working together always beat or make a better decision than a man or a machine independently.”\xa0In future, the three sectors of our country like agriculture sector, industrial sector and service sector are going to utilize the machines. So, that their work becomes not difficult. As of now, we can only see that for agriculture purposes various kinds of machines are used which we called as a modern farming method. Some major technologies [machines] that are harvest automation, autonomous tractors, seeding, and weeing and drones. As a result, farms can do agriculture peacefully. In the industrial sector also humans and machines are working together to increase production. Various types of machines are used in industries such as packing machines, loading machine etc. humans provide instructions to the machines and maintain the management in the company. Soon robots [machines] will assist doctors with surgeries. For instance, a doctor at remote location could direct a surgical robot to perform an open heart surgery. But the approaches option and decision will be left to experience and wisdom of the doctor not the robot.What do you think of machines if they will make humans less or more in the field? Machines will push human professionals up the skillset ladder into uniquely human skills such as creativity, social abilities, empathy, and sense-making, which machines cannot automate. As a result, machines will make the workplace more, not less for humans. However, humans have to learn new skills throughout their lives. It is said that in the future 80% of process-oriented tasks will be done by machines. Quantitative reasoning tasks will be done approximately 50% by humans and 50% by machines, while humans will continue to do more than 80% of cross-functional reasoning tasks. According to Harvard research machines, algorithms can read diagnostic scans with 92% accuracy. Humans can do it with 96% accuracy. Together, it will be 99% accurate.Human-machine collaboration enables companies to interact with employees and customers in the novel, more effective ways. Smart machines are helping humans to expand their abilities in three ways. They can amplify our cognitive strengths; interact with customers and employees to free us from higher-level tasks, and embody human skills to extend our physical capabilities. In the research, it was found that 1,500 companies achieve the most significant performance improvement when humans and machines work together. New machine systems have beyond-human cognitive abilities, which many of us fear could potentially dehumanize the future of work. Machines will indeed automate most repetitive and physical tasks, and part of quantitative tasks such as programming and even data science. According to D.E Shaw Group and professor at the University of Washington, explained that, as a result of machines, there will be more demand for existing jobs, and new jobs will be created that are unimaginable today. This is similar to how we couldn’t imagine a web app developer decades ago, and now millions make a living doing that today.Machines are good at doing tasks with speed, precision, and accuracy. But machines are not very good at responding to unknown situations or making judgments. That part will be left to humans. Hence, the need for both humans and machines will be there in the future. Humans and machines have divergent skill sets that, when combined can transform the way we work. Machines have already infiltrated every aspect of our lives, and we must learn to live with them. In the future, human workers will interact more closely with humans.'
sia = SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']

# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[6]:


#8
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text = 'Machine learning techniques may have been used for years, but recently there has been an explosion in their applications. In fact, in a recent Q3 earnings call, Google CEO Sundar Pichai said “Machine learning is a core, transformative way by which we’re re-thinking how we’re doing everything.” And they’re far from the only business making that claim.In the past, successful use of machine learning algorithms required bespoke algorithms and huge R&D budgets, but all that is changing. IBM Watson, Microsoft Azure, Amazon, and Alibaba all launched turnkey cloud-based machine-learning SaaS solutions in 2015. At the same time startups like Idibon, MetaMind, Dato, and MonkeyLearn have built machine learning products that companies can take advantage of.Gartner already puts machine learning at the top of its hype curve, and no: machine learning won’t replace all of your employees with computers or suddenly double your revenue. But that doesn’t mean that it can’t give every business a competitive advantage. There are plenty of business processes that can significantly benefit from machine learning. So how does machine learning change the way businesses operate?Fig: Machine learning techniques For BusinessBigger upfront costsFirst thing’s first: Machine learning needs training data and training data costs money. Especially training data labeled by humans. Let me explain. To make machine learning work for business, the algorithm needs to see lots and lots of examples of what it’s supposed to be doing. If you want an algorithm to tell you if a sales lead is good, you need to show it lots and lots of examples of good sales leads and bad sales leads. If you want an algorithm to tag the support tickets you need to show it many examples of support tickets. If you localize your algorithm to a new language you probably need to collect lots of examples in that language. In some instances, a company may have those training sets in-house. For example, a bunch of disqualified or qualified leads. But say you haven’t labeled each of your support tickets as they’ve come in over the year. You’d need to have people either in-house or en masse via a data enrichment platform -label those tickets. The machine will then look at those judgments and start finding connections and patterns it can learn from.Much lower ongoing costsMachine learning is much cheaper and more efficient than people when it works well. The downside is that it often works well in 80 percent of the cases and badly in 20 percent of the cases, and lowering the 20 percent error rate is hard, if not impossible. But even an 80 percent accurate algorithm can save you a lot of money because good machine learning algorithms know where they are accurate and where they are more likely to have errors. Smart companies take the cases where the algorithm has high confidence and uses those directly while sending low confidence cases to humans. Banks have been doing this for years. When you put a check in an ATM, an algorithm tries to decipher the numbers on the check. If you have really sloppy handwriting or the ink is smudged the algorithm passes the task to a human. This design pattern saves banks lots of money while preserving a very high level of accuracy.Your costs will drop over timeA huge benefit of machine learning is that it can turn part of your variable cost into more of a fixed cost. If you use humans to handle cases where that algorithm is struggling, you are creating the perfect training data to feed into your algorithm. This is a well-studied technique called active learning it turns out that training data labels collected on cases where the algorithm has low confidence help the algorithm learn much, much more efficiently.As the algorithm becomes increasingly more accurate, the unit economics of your business process become better and as machine learning becomes able to handle more cases, the expensive humans are only called in on the toughest, rarest situations. That means you use the best of both human and machine intelligence in tandem: leveraging the speed and reliability of computers for the easy judgments and the fluency and expertise of humans for the difficult ones. And if that sounds like smart business, it’s because it is.Blackcoffer Insights 28: Monica V, SNS COLLEGE OF TECHNOLOGY'
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']

# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[7]:


#9
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text = 'eLearning as technology becomes more affordable in higher education but having a big barrier in the cost of developing its resources. Deep learning using artificial intelligence continues to become more and more popular and having impacts on many areas of eLearning. It offers online learners of the future intuitive algorithms and automated delivery of eLearning content through modern LMS platforms. This paper aims to survey various applications of deep learning approaches for developing the resources of the eLearning platform, in which predictions, algorithms, and analytics come together to create more personalized future eLearning experiences. In addition, deep learning models for developing the contents of the eLearning platform, deep learning framework that enable deep learning systems into eLearning and its development, benefits & future trends of deep learning in eLearning, the relevant deep learning-based artificial intelligence tools, and a platform enabling the developer and learners to quickly reuse resources are clearly summarized. Thus, deep learning has evolved into developing ways to re-purpose existing resources that can mitigate the expense of content development of future eLearning.It is natural to wonder where you might get AI tools to avoid the time and expense of developing your own. Don’t worry about the advert of AIaaS or “AI as a Service” even small education or learning & development professionals can purchase the license of AI tools and components. However, such types of tools cannot be useful for every e-learning ecosystem but may offer some enticing benefits such as adding standard AI tasks (logic, decision making) to your toolbox. Here are some of the AIaaS tools and platforms offered by famous tech giants most of which are cloud-based.Microsoft AzureCloud-based AI services that can be used to build and manage AI applications like image recognition or bot-based appsIBM’s WatsonCloud-based AI services that can be integrated into your applications; to store and manage your own dataGoogle’s Tensor Flow\xa0An end-to-end open-source machine learning platformAmazon Web ServicesOffers a wide range of products and services on Amazon’s cloudThere are other AIaaS platforms such as DataRobot, Petuum, and H2O which shows that the field is expanding.AI will probably not make human workers obsolete, at least not for a long time To put some of your fears to bed: the robots are probably not coming for your jobs, at least not yet. Given how artificial intelligence has been portrayed in the media, in particular in some of our favorite sci-fi movies, it’s clear that the advent of this technology has created fear that AI will one day make human beings obsolete in the workforce. After all, as technology has advanced, many tasks that were once executed by human hands have become automated. It’s only natural to fear that the leap toward creating intelligent computers could herald the beginning of the end of work as we know it. But, I don’t think there is any reason to be so fatalistic. A recent paper published by the MIT Task Force on the Work of the Future entitled “Artificial Intelligence And The Future of Work,” looked closely at developments in AI and their relation to the world of work. The paper paints a more optimistic picture.Rather than promoting the obsolescence of human labor, the paper predicts that AI will continue to drive massive innovation that will fuel many existing industries and could have the potential to create many new sectors for growth, ultimately leading to the creation of more jobs. While AI has made major strides toward replicating the efficacy of human intelligence in executing certain tasks, there are still major limitations. In particular, AI programs are typically only capable of “specialized” intelligence, meaning they can solve only one problem, and execute only one task at a time. Often, they can be rigid, and unable to respond to any changes in input, or perform any “thinking” outside of their prescribed programming. Humans, however, possess “generalized intelligence,” with the kind of problem-solving, abstract thinking, and critical judgment that will continue to be important in business. Human judgment will be relevant, if not in every task, then certainly throughout every level across all sectors. There are many other factors that could limit runaway advancement in AI. AI often requires “learning” which can involve massive amounts of data, calling into question the availability of the right kind of data, and highlighting the need for categorization and issues of privacy and security around such data. There is also the limitation of computation and processing power. The cost of electricity alone to power one supercharged language model AI was estimated at $4.6 million. Another important limitation of note is that data can itself carry bias, and be reflective of societal inequities or the implicit biases of the designers who create and input the data. If there is bias in the data that is inputted into an AI, this bias is likely to carry over to the results generated by the AI.There has even been a bill introduced into Congress entitled the Algorithmic Accountability Act with the goal of forcing the Federal Trade Commission to investigate the use of any new AI technology for the potential to perpetuate bias. Based on these factors and many others, the MIT CCI paper argues that we are a long way from reaching a point in which AI is comparable to human intelligence, and could theoretically replace human workers entirely.\xa0 Provided there is an investment at all levels, from education to the private sector and governmental organizations—anywhere that focuses on training and upskilling workers—AI has the potential to ultimately create more jobs, not less. The question should then become not “humans or computers” but “humans and computers” involved in complex systems that advance industry and prosperity. This paper is a fascinating read for anyone hoping to dive deeper into AI and the many potential directions in which it may lead.AI Is becoming standard in all businesses, not just in the world of tech A couple of times recently, AI has come up in conversation with a client or an associate, and I’m noticing a fallacy in how people are thinking about it. There seems to be a sense for many that it is a phenomenon that is only likely to have big impacts in the tech world. In case you hadn’t noticed, the tech world is the world these days. Don’t ever forget when economist Paul Krugman said in 1998 that “By 2005 or so, it will become clear that the Internet’s impact on the economy has been no greater than the fax machine’s.” You definitely don’t want to be behind the curve when it comes to AI. \xa0In fact, 90% of leading businesses already have ongoing investments in AI technologies. More than half of businesses that have implemented some manner of AI-driven technology report experiencing greater productivity. AI is likely to have a strong impact on certain sectors in particular:Medical:The potential benefits of utilizing AI in the field of medicine are already being explored. The medical industry has a robust amount of data, which can be utilized to create predictive models related to healthcare. Additionally, AI has shown to be more effective than physicians in certain diagnostic contexts.Automotive:We’re already seeing how AI is impacting the world of transportation and automobiles with the advent of autonomous vehicles and autonomous navigation. AI will also have a major impact on manufacturing, including within the automotive sector.Cybersecurity:Cybersecurity is front of mind for many business leaders, especially considering the spike in cybersecurity breaches throughout 2020. Attacks rose 600% during the pandemic as hackers capitalized on people working from home, on less secure technological systems, and Wi-Fi networks. AI and machine learning will be critical tools in identifying and predicting threats in cybersecurity. AI will also be a crucial asset for security in the world of finance, given that it can process large amounts of data to predict and catch instances of fraud.E-Commerce:AI will play a pivotal role in e-commerce in the future, in every sector of the industry from user experience to marketing to fulfillment and distribution. We can expect that moving forward, AI will continue to drive e-commerce, including through the use of chat-bots, shopper personalization, image-based targeting advertising, and warehouse and inventory automation.AI can have a big impact on the job searchIf you are moving forward with the hope that a hiring manager may give you the benefit of the doubt on a small misstep within the application, you might be in for a rude awakening. AI already plays a major role in the hiring process, so much so that up to 75% of resumes are rejected by an automated applicant tracking system, or ATS before they even reach a human being.\xa0 In the past, recruiters have had to devote considerable time to poring over resumes to look for relevant candidates. Data from LinkedIn shows that recruiters can spend up to 23 hours looking over resumes for one successful hire.Increasingly, however, resume scanning is being done by AI-powered programs. In 2018, 67% of hiring managers stated that AI was making their jobs easier. Despite the increasing prevalence of automation and algorithms in the hiring process, many have been critical of the use of certain types of AI by hiring managers, based on the charge that it can perpetuate and ever create more bias in hiring. One particular example is illustrated by HireVue, a startup whose initial services included technology that aimed to use facial recognition software and psychology to determine the potential effectiveness of a candidate in a certain role. The Electronic Privacy Information Center filed a lawsuit with the Federal Trade Commission alleging that this software had the potential to perpetuate bias and prejudice. HireVue discontinued the use of facial recognition software in early 2021, and now uses audio analysis and natural language processing. It’s clear that the use of certain types of AI in the hiring process will likely be controversial as new technology develops. However, if potential employers are using AI to process your application, there is no reason that you cannot be utilizing similar technology to your advantage.Jobscan is an excellent resource that provides similar resume scanning to what would be used by a hiring manager. By comparing your resume to a job description, Jobscan will give you information on how to tweak your resume so that it is a good match for a certain position, with the goal of “beating” an applicant tracking system (ATS).Jobseer is a browser add-on, and another great AI-based tool for those on the job market. Based on a scan of your resume, as well as keywords and skills related to your desired jobs, Jobseer will help match you with the job listings that best fit your experience. For each listing, you get a rating based on how well you are aligned with the particular posting, as well as recommendations of skills to add to better position your resume and experience.Rezi: Now, as a disclaimer, I would never encourage you to turn your resume writing over to a bot. But Rezi is an awesome AI-based resume builder that includes templates to help you design a resume that is sure to check the boxes when it comes to applicant tracking systems. This is a great jumping-off point to kickstart a new resume.\xa0 Another great way to use this type of tool is to generate a new resume and compare it to your current resume to see how it stacks up, and identify some areas for improvement. AI is also a great place to focus your energy if you are looking to upskill in your career, or make your professional profile more competitive in the job market, especially when you consider that AI will have such far-reaching impacts across many industries.AI and machine learning are at the top of many lists of the most important skills in today’s job market. Jobs requesting AI or machine-learning skills are expected to increase by 71% in the next five years. If you’d like to expand your knowledge base in this arena, consider some of the great free online course offerings that focus on AI skills. If you are tech-savvy, it would be wise to dive deep and learn as much as you can about interacting in the AI space. If your skills lie elsewhere, it is important to recognize that AI will have a big impact, and to the extent of your abilities, you should try to understand the fundamentals of how it functions in different sectors. AI is definitely here to stay, whether we like it or not. Personally, I don’t think we have anything to be afraid of. The best way to move forward is to be aware of and adapt to the new technology around us, AI included. This article was updated on April 16, 2021, to reflect changes in HireVue’s assessment tools.Blackcoffer Insights 28: Monica V, SNS COLLEGE OF TECHNOLOGY'
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']

# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[8]:


#1
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text ='Introduction“If anything kills over 10 million people in the next few decades, it will be a highly infectious virus rather than a war. Not missiles but microbes.” Bill Gates’s remarks at a TED conference in 2014, right after the world had avoided the Ebola outbreak. When the new, unprecedented, invisible virus hit us, it met an overwhelmed and unprepared healthcare system and oblivious population. This public health emergency demonstrated our lack of scientific consideration and underlined the alarming need for robust innovations in our health and medical facilities. For the past few years, artificial intelligence has proven to be of tangible potential in the healthcare sectors, clinical practices, translational medical and biomedical research.After the first case was detected in China on December 31st 2019, it was an AI program developed by BlueDot that alerted the world about the pandemic. It was quick to realise AI’s ability to analyse large chunks of data could help in detecting patterns and identifying and tracking the possible carriers of the virus.Many tracing apps use AI to keep tabs on the people who have been infected and prevent the risk of cross-infection by using AI algorithms that can track patterns and extract some features to classify or categorise them.So how does AI do that?IBM Watson, a sophisticated AI that works on cloud computing and natural language processing, has prominently contributed to the healthcare sector on a global level. Being a conversational AI, since 2013, Watson has helped in recommending treatments to patients suffering from cancer to ensure that they get the best treatment at optimum costs.Researchers at Google Inc. showed that an AI system can be trained on thousands of images to achieve physician-level sensitivity.By identifying the molecular patterns associated with disease status and its subtypes, gene expression, and protein abundance levels, machine learning methods can detect fatal diseases like cancer at an early stage. Machine Learning (ML) techniques focus mainly on analyzing structured data, which can further help in clustering patients’ traits and infer the probability of disease outcomes. Since patient traits mainly include masses of data relating to age, gender, disease history, disease-specific data like diagnostic imaging and gene expressions, etc, ML can extract features from these data inputs by constructing data analytical algorithms.ML algorithms are either supervised or unsupervised. Unsupervised learning helps in extracting features and clustering similar features together that further leads to early detection of diseases. Clustering and principal component analysis enable grouping or clustering of similar traits together that are further used to maximize or minimize the similarity between the patients within or between the clusters. Since patient traits are recorded in multiple dimensions, such as genes, principal component analysis(PCA) creates the apparatus to reduce these dimensions which humans could have not done alone.Supervised learning considers the outcomes of the subjects together with the traits, and further correlates the inputs with the outputs to predict the probability of getting a particular clinical event, expected value of a disease level or expected survival time, or risk of Down’s syndrome.Biomarker panels that are mostly used to detect ovarian cancer, have outperformed the conventional statistical methods due to machine learning. In addition to this, the use of EHRs and Bayesian networks, which are a part of supervised machine learning algorithms, can predict clinical outcomes and mortality respectively.Unstructured data such as clinical notes and texts are converted into machine-readable structured data with the help of natural language processing(NLP). NLP works with two components: text processing and classification. Text processing helps in identifying a series of disease-relevant keywords in clinical notes and then through classification are further categorized into normal and abnormal cases. Chest screening through ML and NLP has helped find abnormalities in the lungs and provide treatment to covid patients. Healthcare organizations use NLP-based chatbots to increase interactions with patients, keeping their mental health and wellness in check.Deep learning is a modern extension of the classical neural network techniques which helps explore more complex non-linear patterns in data, using algorithms like convolution neural network, recurrent neural network, deep belief network, and deep neural network which enables more accurate clinical prediction. When it comes to genome interpretation, deep neural networks surpass the conventional methods of logistics regression and support vector machines.Sepsis Watch is an AI system trained in deep learning algorithms that holds the capability to analyze over 32 million data points to create a patient’s risk score and identify the early stages of sepsis.Another method known as the Learning-based Optimization of the Under Sampling Pattern( LOUPE) is based on integrating full resolution MRI scans with the convolutional neural network algorithm, which helps in creating more accurate reconstructions.Robotic surgery is widely considered in most delicate surgeries like gynaecology and prostate surgery. Even after striking the right balance between human decisions and AI precision, robotic surgery reduces surgeon efficiency as they have to be manually operated through a console. Thus, autonomous robotic surgery is on the rise with inventions such as robotic silicon fingers that mimic the sense of touch that surgeons need to identify organs, cut tissues, etc., or robotic catheters that can navigate whether it is touching blood, tissue, or valve.Researchers at Children’s National Hospital, Washington have already developed an AI called Smart Tissue Autonomous Robot (STAR), which performs a colon anastomosis on its own with the help of an ML-powered suturing tool, that automatically detects the patient’s breathing pattern to apply suture at the correct point.An image of STAR during surgery.Cloud computing in healthcare has helped in retrieving and sharing medical records safely with a reduction in maintenance costs. Through this technology doctors and various healthcare workers have access to detailed patient data that helps in speeding up analysis ultimately leading to better care in the form of more accurate information, medications, and therapies.How can It help in Biomedical research?Since AI can analyze literature beyond readability, it can be used to concise biomedical research. With the help of ML algorithms and NLP, AI can accelerate screening and indexing of biomedical research, by ranking the literature of interest which allows researchers to formulate and test scientific hypotheses far more precisely and quickly. Taking it to the next level, AI systems like the computational modelling assistant (CMA) helps researchers to construct simulation models from the concepts they have in mind. Such innovations have majorly contributed to topics such as tumour suppressor mechanisms and protein-protein interaction information extraction.AI as precision medicineSince precision medicine focuses on healthcare interventions to individuals or groups of patients based on their profile, the various AI devices pave the way to practice it more efficiently. With the help of ML, complex algorithms like large datasets can be used to predict and create an optimal treatment strategy.Deep learning and neural networks can be used to process data in healthcare apps and keep a close watch on the patient’s emotional state, food intake, or health monitoring.\xa0“Omics” refers to the collective technologies that help in exploring the roles, relationships of various branches ending with the suffix “omics” such as genomics, proteomics, etc. Omics-based tests based on machine learning algorithms help find correlations and predict treatment responses, ultimately creating personalized treatments for individual patients.\xa0How it helps in psychology and neuro patientsFor psychologists studying creativity,\xa0 AI is promising new classes of experiments that are developing data structures and programs and exploring novel theories on a new horizon. Studies show that \xa0AI can conduct therapy sessions, e-therapy sessions, and assessments autonomously, also assisting human practitioners before, during, or after sessions. The Detection and Computational Analysis of Psychological Signal project uses ML, computer vision, and NLP to analyze language, physical gestures, and social signals to identify cues for human distress. This ground-breaking technology assesses soldiers returning from combat and recognizes those who require further mental health support. In the future, it will combine data captured during face-to-face interviews with information on sleeping, eating, and online behaviours for a complete patient view.Stroke identificationStroke is another frequently occurring disease that affects more than 500 million people worldwide. Thrombus,\xa0 in the vessel cerebral infarction is the major (about 85%) cause of stroke occurrence. In recent years, AI techniques have been used in numerous stroke-related studies as early detection and timely treatment along with efficient outcome prediction can help solve the problem. With AI at our disposal, large amounts of data with rich information, more complications and real-life clinical questions can be addressed in this arena. Currently, two ML algorithms- genetic fuzzy finite state machine and PCA were implemented to build a model building solution. These include a human activity recognition stage and a stroke onset detection stage. An alert stroke message is activated as soon as a movement significantly different from the normal pattern is recorded. ML methods have been applied to neuroimaging data to assist disease evaluation and predicting stroke treatment for the diagnosis.Patient MonitoringToday, the market for AI-based patient monitoring is impressive and monetarily enticing. It is evolving with artificial sensors, smart technologies and explores everything from brain-computer interfaces to nanorobotics. Companies with their smart-watches have engaged people to perform remote monitoring even when they are not “patients”. An obvious place to start is with wearable and embedded sensors, glucose monitors, pulse monitors, oximeters, and ECG monitors. With patient monitoring becoming crucial, AI finds numerous applications in chronic conditions, intensive care units, operating rooms, emergency rooms, and cardiac wards where timeless clinical decision-making can be measured in seconds. More advances have started to gain traction like smart prosthetics and implants. These play an impeccable role in patient management post-surgery or rehabilitation. Demographics, laboratory results and vital signs can also be used to predict cardiac arrest, transfer into the intensive care unit, or even death. In addition, an interpretable machine-learning model can assist anesthesiologists in predicting hypoxaemia events during surgery. This suggests that with deep-learning algorithms, raw patient-monitoring data could be better used to avoid information overload and alert overload while enabling more accurate clinical prediction and timely decision-making.\xa0ConclusionConsidering the vast range of tasks that an AI can do, it is evident that it holds deep potential in improving patient outcomes to skyrocketing levels. Using sophisticated algorithms AI can bring a revolution in the healthcare sector. Even after facing challenges like whether the technology will be able to deliver the promises, ethical measures, training physicians to use it, standard regulations etc, the role of AI in transforming the clinical practices cannot be ignored. The biggest challenge is the integration of AI in daily practice. All of these can be overcome and within that period the technologies will mature making the system far more enhanced and effective.'
negative_score = sentiment_scores['neg']

# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[16]:


#2
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text=' Human minds, a fascination in itself carrying the potential of tinkering nature with the pixie dust intelligence, creating and solving the mysteries and wonders with anything but admiration. However, no matter how captivating a human mind can be, it could sometimes be appalled. It could be the hunger or maybe the desire to want more, to go beyond and unravel the limitations, or maybe something like pure greed. Humans have never stopped and always keep evolving when it comes to intelligence and this is what makes them the supreme. Intelligence calls out for supremacy and so, what if there was to evolve something that opposed a challenge to the very human minds, to their capabilities while making them question their own importance among themselves? Artificial Intelligence came as a revolution, havoc when it first came to the light. The concept of making machines does work on their own, like granting machines –The Intelligence. The idea of making machines work like humans came back in the 19s. Back then people didn’t believe in such a thing as making a non-living thing work, think, and carry tasks on its own, not to mention, to actually surpass humans themselves in those skills. The facts are it did. By 1997. The greatest chess player, Garry Kasparov was defeated in a chess game by a machine and this is where exactly, a top skilled human lost to a mere machine created by another who by himself could’ve never defeated him. It was a rule of power, of betterment, of skills, and the granted supremacy. Were AI and Machines just tools? Equipment?\xa0 Something that helped an unskilled person with his mind and intelligence creates something that could do the skilled work for him with perfection and precision? Well initially it was, however, as time passed as humans got drawn to the puzzle of AI, a lot changed. Human research went deeper and deeper and as a result, the machines evolved with it. At present, AI & Machines is a growing field. As it develops and improves, it has become a part of the industrial revolution. In industries, most of the laborious work that was once taken care of by humans was now replaced by machines. Naturally, with the evolution in machines, its precision, mass productivity, quality control, time efficiencies, and all the other factors made it a better choice. A choice over humans. This led to fear, a fear of a not-so-distant future, a future where maybe machines will be so evolved that they’ll take over the need of a human employee leading to unemployment. With the population increase around the world, it became the new tech threat for the labor market. Then again… how true is it? Does AI really oppose a threat? Will adapting to technology make millions of people lose their jobs? Will it lead to mass unemployment? Will the machines really surpass humans? Will, the creation take over the creator? No matter how fearful the future with AI may seem, in reality, it is not that scary. Truth is AI is the present reality, it is the key that holds the power to unlock a whole next level of human evolution. Technology is growing. There was a time where technology was just an idea, but today that idea has been implemented, it’s working and is carried out. Nobody could stop the advancement and growth of Artificial Intelligence, it’s a wave that is already flowing and we as the present generation and the generations to come to have to learn, to learn to swim in this flow and avoid drowning. Many jobs will be replaced by machines, as AI evolves it’ll keep challenging human minds and their skills. With the present COVID 19 situation, contactless cashiers to robots delivering packages have already taken over the usual routine tasks. The jobs of Secretaries, Schedulers, and book-keeper are at risk too. Manufacturing units, agriculture, food services, retail, transportation & logistic, and hospitality are all a part of the AI-affected automation. At an estimation, it is said that around 20 million jobs, especially including manufacturing will be lost to robots. As AI, robotics, 3D printing, and genetics make their way in, even the architects, medical docs, and music composers feel threatened by technology. Making us question that will AI even edge us out of our brain jobs too? Now that can be terrifying. However, as much as machines will be replacing few jobs, they’ll also be creating new jobs.\xa0 With the economic growth, innovation, and investment around 133 million jobs are said to be generated. These newly enhanced jobs are to create benefits and amplify one’s creativity, strategy, and entrepreneurial skills. So what is the catch? Well, it’s the skills. Even though AI is creating 3 times more jobs than it is destroying, it’s the skills that count. AI surged in new job opportunities, opportunities like Senior Data Scientist, Mobile Application Developer, and SEO specialist. These jobs were once never heard of but now with AI it’s born, however, to do these jobs or for its qualification, one needs high-level skills and to acquire those skills can be an expensive and time-consuming task. The future generation might be able to cope up with it but the real struggle is to be faced by the present two generations. It’s the vulnerability between the skill gap and unemployment and the youths are the ones to be crushed the most. Therefore, as the advancement of AI becomes inevitable there remains no choice but to adapt, learn, equip ourselves and grow with it. The companies have to work together to build an AI-ready workplace. They should collaborate with the government, educators, and non-profit organizations and work together to bring out policies that could help understand the technologies’ impacts faster while also providing the employees some security. The economic and business planning should be made considerable for minimizing the impact on local jobs and properly maximizing the opportunities. The employees should be provided with proper tools to carry along with the new opportunities while acquiring AI-based skills for their day-to-day work. New skills should be identified and implemented for the upskilling and continual learning initiatives. Employees will have to maximize their Robotic Quotient and learn core skills. They’ll have to adapt to new working models and understand their roles in the coming future.\xa0 Howsoever, it’s not like AI will totally take over control, even though AI proves to be a better choice, it still has its limitations at present. First, it’s expensive, secondly, manufacturing machines in bulk is not good for the environment. Machines are also very high maintenance, therefore human labor will often come cheaper and so will be considered over machines. Underdeveloped countries will find it hard to equip their people with the upskilling and reskilling required for AI workplace and so for AI to play a role in those countries, might take years. AI can also be risky and unethical, as it’s hard to figure out who to be held responsible for in cases where an AI went wrong. No matter, how advanced AI gets, there are some skills where humans will always have an upper hand i.e., soft skills. Skills like teamwork, communication, creativity, and critical thinking are something that AI hasn’t been able to beat us up to yet and so the value of creativity, leadership, and emotional intelligence has increased. Although, with machines coming in between humans causing the lack of human-to-human interaction, the humans seem to fade away a little. With this era, comes the need for good leaders. Leaders who are capable of handling both machines and humans together, the ones who are organized enough to manage the skilled and the unskilled employees while providing the unskilled trainees with proper training. Leaders who hold profound soft skills and encourage teamwork while working along with machines. The ones who are patient, calm, and optimized. \xa0 In conclusion, yes AI and machines are going to be very challenging but there’s nothing humans haven’t overcome. Adaptation and up-gradation are going to be the primary factor for survival. As we witness the onset of the 4th industrial revolution, let’s buckle up our seats and race along the highway with the essential fuels (skills) so as to not let ourselves eliminated. After all, this is an unending race with infinity as the end, all we could do is try not to run out of fuel. Try not to be outdated.\xa0 Blackcoffer Insights 29: Glady, Karunya Institute of Technology and Sciences.  '
sia = SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[14]:





# In[2]:


#3
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text=' Introduction AI is rapidly evolving in the employment sector, particularly in matters involving business and finance. Finance, management, economics, and accounting are now among the most popular university courses globally; particularly at the graduate level, due to their high employability. However, the evolution of machinery in industries is changing that. According to research, 230,000 jobs in these sectors may be replaced by AI agents in the next 5 years. This is due to the nature of the work, as employees are responsible for tasks such as data analysis and keeping track of numerical information; which machines excel at. Large, complicated data sets can be analyzed faster and more efficiently by AI-powered computers than by people. Algorithmic trading procedures that produce automated deals are less likely to be produced without minute errors when undertaken by humans. In such matters involving industrial work, a subsection of artificial intelligence is used; namely, machine learning.  Machine learning is a term used for the application of artificial intelligence (AI) in systems, which involves them assimilating and processing information by gaining experience. It is mainly concerned with the development of technology and computer programs. \xa0Its improvement process is mainly divided into seven steps; which start with the collection of data. This data can be collected from an internal database or perhaps an IoT structure. The second and most time-consuming step is cleaning, preparing, and rearranging the data; which involves the recognition of outliers, trends, and missing information so that the outcome is as accurate as possible. The third step consists of formatting data; which is useful if you source your information from a variety of sources. Step four is where AI comes into place. Self-service data processing tools may be useful if they provide intelligent services for matching data attributes from distinct databases and intelligently integrating them. The data is then arranged to better represent a specific pattern. Lastly, the data is divided into two sets: one for training the algorithm and one for analysis. There are three types of machine learning; supervised, unsupervised, and reinforcement learning. The most common of the bunch is supervised machine learning; which is based on accurately labeled data. The machine is given a collection of information, including the outcome of the operation. The rest of the information is referred to as ‘input features’, which ‘supervises’ the machine by guiding it to establish the connections between the variants. In the case of unsupervised learning, the machine isn’t given labeled sets of data to be divided into, allowing it to recognize and create its own patterns within the information provided. Since computers have the capabilities of identifying distinguished similarities, this method helps with the classifying of such data. Reinforcement learning comprises experience-based learning. Similar to people, they learn due to a reward and punishment system based on their actions. These variations of system learning have recently been integrated into business and finance, which is elaborated on in the following passages. Machine learning in finance and banking There are many ways in which machine learning is used in finance and banking. The most common place it is used is to detect any frauds. Fraud is the most common problem in financial service companies and it accounts for billions of dollars in misfortune each year. The most common frauds are credit or debit card usage by a stranger, document forgery, and mortgage fraud. Usually, finance companies keep an enormous amount of their data stored online, and it increases the risk of information being accessed without authorization. With expanding innovative headway, misrepresentation in the monetary business is currently viewed as a high danger to significant information. Fraud recognition frameworks in the past were planned dependent on a bunch of rules, which could be effortlessly be bypassed by current fraudsters. Therefore, most companies today leverage machine learning to combat fraudulent financial transactions. \xa0Machine learning works by looking over enormous informational indexes to distinguish unique or suspicious activities for additional examination by security groups. It works by looking at an exchange against other information focuses – like the client’s record history, IP address, area, and so forth.\xa0 Depending on the type of exchange taking place, the program can automatically refuse a withdrawal or purchase until the person makes a decision. Examples of fraud detection software used by banks include Feedzai, Data visor, and Teradata Machine learning is also used in algorithmic trading, portfolio management, loan underwriting, chatbots, and improving customer service. Algorithmic trading is an interaction for executing orders using computerized and pre-modified exchanging guidelines to represent factors like value, timing, and volume. An algorithm is a set of directions for solving a problem. Computer calculations send little parts of the full request to the market over the long run. In contrast to human dealers, algorithmic exchanging can investigate enormous volumes of information every day and therefore make thousands of trades every day. Machine learning settles on quick exchanging choices, which gives human traders a benefit over the market normal. Likewise, algorithmic exchanging doesn’t settle on exchanging choices dependent on feelings, which is a typical constraint among human dealers whose judgment might be influenced by feelings or individual desires. The exchanging technique is generally utilized by multifaceted investment administrators and monetary foundations to automate trading activities. In the banking industry, organizations access a large number of shopper information, with which machine learning can be prepared to work in order to simplify the underwriting process. Machine learning calculations can settle on speedy choices on endorsing and credit scoring, and save organizations both time and monetary assets that are utilized by people. Data scientists can train algorithms on how to analyze millions of consumer data to coordinate with information records, search for interesting special cases, and settle on a choice on whether a shopper meets all requirements for an advance or protection. For instance, the calculation can be prepared on the most proficient method to examine shopper information, like age, pay, occupation, and the buyer’s credit conduct. The benefits of machine learning As with any form of revolutionary technology, the usage of machine learning has been debated over as its beneficial properties have been weighed against the possible disadvantages. It’s been observed that upon correct usage, it may be used to solve a wide range of business challenges and anticipate complicated customer behavior. We’ve also seen several of the largest IT conglomerates, such as Google, Amazon, and Microsoft, introduce Cloud Machine Learning platforms. Technology vs Human intelligence In terms of business, ML can aid businesses by identifying consumer’s demands and formulate a pattern based on them; allowing companies to reach out to such consumers and maximize their sales. It eliminates the need for expenses on some maintenance by reducing the risks associated with unexpected breakdowns and minimizes wasteful costs to the firms.\xa0 Historical data, a process visualization tool, a flexible analytical environment, and a feedback loop may all be used to build an ML architecture. The input of data is another agitating chore for businesses in the present, which is where machine learning can step in for manual labor. This eradicates the possibility of errors caused by manual labor causing disruptions and provides employees with extra time to handle other tasks. In the financial sector, machine learning is already utilized for portfolio management, algorithmic trading, loan underwriting, and fraud detection. Chatbots and other conversational interfaces for security, customer support, and sentiment analysis will be among the future uses of ML in banking. Another aspect of business involving artificial intelligence includes image recognition, which entails a system or program that recognizes objects, people, places, and movements in photos. It identifies photographs using an imaging system and machine recognition tech with artificial intelligence and programmed algorithms. The downfalls of machine learning With all the benefits of its advancement, machine learning isn’t the most perfect thing. There are several disadvantages which are information acquisition, time and resources and high errors, and wrong interpretations. One of the major hurdles is the amount of finance needed to invest in machine learning for it to be a successful project. More issues have to do with the fact that AI requires gigantic informational indexes to train on, and these ought to be unbiased, and of good quality. There can likewise be times where they have to wait for that new information to be produced. Machine learning needs sufficient opportunity to do the calculations to learn and adequately to satisfy their accuracy and relevancy. It also needs huge resources to work. This can mean extra requirements for the computer to work. Another significant problem is the capacity to precisely decipher the results produced by the calculations. You should likewise cautiously pick the algorithms to get the wanted results. Conclusion There have been various reports in the past and current years which claim that a significant piece of the human labor force will be replaced via robots and machines in the years to come. With excessive innovative work being led in the field of computerized reasoning, many dread that a significant job crisis will unfurl since numerous positions are all the more precisely and productively performed with the use of machines. In countries like Japan, mainly computer programs and AI is used in the secondary and tertiary sectors. From cleaning the house to depositing money in banks, everything is done by AI. However, AI cannot replace humans in the future. Humans have several capabilities which, even after several technological advancements a machine would not be able to have. These capabilities include creative thinking and creative problem solving, and human connection. For example, when a child goes to a doctor to get an injection, a nurse always relaxes the child to not be afraid of the needle. A machine’s touch would not be able to soothe a child. Another example could be how humans tend to share things with each other and be open about it, a machine will not be able to do so since it is only programmed to things it has been told to do. Like computers, AI will not replace us but would however complement us to make daily work easier and less time-consuming. Without humans themselves, there is no future for AI. Blackcoffer Insights 29: Amara Arora and Vaanya Kaushal, Scottish High International School '
sia = SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[3]:


#4
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='“Anything that could give rise to smarter-than-human intelligence – in the form of Artificial Intelligence, brain-computer interfaces, or neuroscience-based human intelligence enhancement – wins hands down beyond contest as doing the most to change the world. Nothing else is even in the same league.”\xa0–Eliezer Yudkowsky, AI ResearcherThere’s no denying robots and automation are increasingly part of our daily lives. Just look around the grocery store, or the highway, they are everywhere. This makes us wonder what if AI can replace human intelligence? What can we do to make ourselves relevant tomorrow? Let us try to find the answers to all these questions and more.Let’s first understand what is Artificial Intelligence –Artificial Intelligence or AI basically machines displaying intelligence. This can be seen from a machine playing chess or a robot answering questions on Facebook. Artificial Intelligence can be further broken down into many different types. There are AIs designed to do specific tasks, such as detecting a specific type of cancer. However, there are also AIs that can do multiple tasks, such as driving a car. There are many types of AIs. Among the top, most important fields are Machine Learning or ML, Neural Network, Computer Vision, and Natural Language Processing or NLP.Machine Learning is the idea of machines being able to prove themselves similar to how a human being learns a new skill. Machine Learning also allows for the optimization of an existing skill. Machine Learning is used in many different fields and one such application is entertainment. Netflix uses Machine Learning to recommend more shows that you can watch based on the shows that you have already seen.Neural Networks are algorithms that are modeled after the human brain. These algorithms think just like we do which can thereby give similar results to what a human being can give. Artificial Neural Networks are used in medical fields to diagnose cancers like lung cancer and prostate cancer.Computer Vision is the idea that computers have visions. This allows them to see things the way human beings do or potentially better than human beings do, depending on the programming, camera used, etc. Computer Vision is used in autonomous vehicles for navigation from one place to another.Natural Language Processing is the idea that computers can listen to what we say. An example of this is Siri. Siri is able to listen to our demands, process what it means, and provide you an answer based on what is researched.Now that we know what an AI is and what it can do, Let’s talk about the issue.WILL MACHINE REPLACE THE HUMAN IN THE FUTURE OF WORK?\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0 \xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0 \xa0\xa0\xa0AIs allow for the automation of jobs, thereby replacing what humans already do. This means more job loss and the concentration of wealth to the selected few people. This could mean a destabilization of society and social unrest. In addition to social unrest, AI improves over time. This means it becomes smarter, faster, and cheaper to implement and it will be better at doing repetitive things that humans currently do, such as preparing fast food. It is predicted that AI will improve so much over 50 to 100 years that AI will become super intelligent. This means that it will become even smarter than the most intelligent human on earth. According to many experts such as Elon Musk, this could cause the end of human civilization. AI could potentially start a war against humans, burn crops and do all sorts of tragedies once reserved for human functions. At that point, in theory, we can not stop it because AI would have already thought of all the obstacles that will prevent its goal. This means that we cannot unplug the machine, in effect AI will replace human intelligence.But, will this happen in next 10 to 30 years?NO! The field of Artificial Intelligence is sophisticated enough to do many human tasks that humans currently do. Currently, AI is not smart enough to be empathetic to humans and cannot think strategically enough to solve complex problems. AI solutions can be expensive and have to go through many different tests and standards to implement. It also takes time for AI to improve. For example, Boston Dynamics, one of the world’s top robotics company had a robot in 2009 that needed assistance to walk. Fast forward to 2019, not only the robot could walk by itself but it could jump over objects, do backflips and so much more. In addition to the timing, it takes time for the price of any new technological solution to drop to a point where it is affordable. For example, a desktop computer costs around $1000 in 1999 but now you can get a significantly more powerful laptop for the exact same price. AI will go through the same curve.But what happens after those 10 to 30 years? Will AI make human intelligence obsolete? Maybe. As we have proven earlier AI will become faster better and cheaper. As this happens, more and more companies will use AI technology to automate more and more jobs to save money, increase productivity, and most importantly, stay competitive. As we have demonstrated, AI will become better through repetition via the use of machine learning. The only difference is that AI will be able to learn faster as time progresses due to the amount of data that is available today. It will also be able to learn from other machines or similar machines to learn how to optimize its tasks or new important skills. However, AI also just not do repetitive and routine tasks better, it will also be able to understand emotional intelligence, ethics, and creativity. This seen in three distinct example- IBMIBM uses its IBM Watson to program the AI to create a movie trailer. Fox approached IBM and said they have a movie coming out on AI #Scifi horror. They asked IBM if their platform IBM Watson could a trailer by reviewing and watching the footage and searching for scary,WILL MACHINE REPLACE THE HUMAN IN THE FUTURE OF WORK?\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0 \xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0Sad or happy or other moments in the movie that provoked quality emotions based on how the machine was programmed to identify such emotions in a quantifiable manner. IBM Watson was able to generate a trailer for the movie Morgan. The result, a movie trailer created by machines example – GoogleIN 2018 google demonstrated an AI assistance that could take calls and do simple stuff. The AI was able to set up an appointment! What was more fascinating was that it was able to understand the nuances of the conversation. The receptionist thought it was a human being that was calling her. That is a very primitive version of what is possible with this technology. Eventually, it will be able to have conversations just as human beings do, making many sales jobs obsolete. example – AI generated artIn 2018, a Paris art collective consisting of three students used artificial intelligence to generate a portrait. It generated the portrait painting by studying a set of fifteen thousand art images on wiki art. It was estimated to be worth between seven thousand to ten thousand dollars. The painting sold at an auction for four thirty-five thousand US dollars.However, we cannot for sure say that AI will replace human intelligence. This is because we as a society have started asking hard questions and questioning ethics. Elon Musk founded Open AI, a research lab whose whole purpose is to promote and discover artificial intelligence in a way to benefits humanity. In addition to this, there are many factors that affect the long-term outcome of AI replacing human intelligence. Like, to what degree will other humans allow for AI to take over? Depending on the field, do people even want Artificial Intelligence to help them? Or will they prefer a human counterpart? While we may not be able to control what happens in the long run, we can definitely secure our short-term future.Here are the top five skills that will not become obsolete in the near futureStrategic and creative thinkingThe ability to think outside the box is very human. There are thousands upon thousands of slightly different possible outcomes that may result from every distinguishable action that the human mind with its ability to judge from experience is programmed for these purposes in a far more sophisticated manner than AI can currently achieve. As the billionaire founder of Alibaba, Jack Ma famously said – “AI has logic, human beings have wisdom”.Conflict resolution and negotiationsWith our understanding of the complexities of human-related processes and our ability to improvise and judge, we are far better equipped to deal with conflicts than robots are ever likely to be.WILL MACHINE REPLACE THE HUMAN IN THE FUTURE OF WORK?\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0 \xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0Emotional Intelligence and EmpathyAI may be able to recognize faces and images but it can rarely successfully read the feelings of those faces. Humans, to lesser or greater degrees, are capable of an accurate analysis of emotional subtext. With the application of intuition and the use of delicately worded or elusive languages, through these methods, we are able to properly judge how a person feels.Interpretation of Gray AreasRobots and computers function well when presented with quantifiable data. However, once the situation enters a gray area, whether this term refers to morals, processes, or definitions robots are more likely to falter.Critical thinkingHumans are capable of responding to more indicators of quality than computers are. While an AI system may be able to analyze documents according to the true or false statements made within the text, we can judge whether or not it is well written and analyze the implication of the use of certain words and the overall meaning of the content.Blackcoffer Insights 29:\xa0Fiza Parveen, Shri Govindram Institute\xa0of Technology and Science, Indore'
sia = SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[4]:


#5
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='“Machine intelligence is the last invention that humanity will ever need to make”Nick BostromTo put it frankly, Artificial Intelligence will eventually replace jobs. Workers in a variety of industries, from healthcare to agriculture and manufacturing, should expect to witness hiring disruptions as a result of Artificial Intelligence.If history has taught us anything, it is that disruptive paradigm-shifting business ideas not only make a fortune for the innovators, but they also build the groundwork for new business models, market entrants, and job opportunities which will inevitably follow. It is true that robots today or in future will eventually replace humans for many jobs, but so did innovative farming equipment for humans and horses during the industrial revolution. But that does not mean that our jobs as humans will end here. We, on the other hand, will be required to generate and provide value in whole new ways for entirely new business models as a result of these changes.According to 71% of the businesses worldwide, Artificial Intelligence can help people overcome critical and challenging problems and live better lives. Artificial Intelligence consultants at work will be more or equally fair, according to a whopping 83% of corporate leaders. These results demonstrate that Artificial Intelligence is steadily extending its measures, yielding societal benefits and allowing citizens to live more fulfilling lives.Increase in Automation and Jobs where humans can’t competeSince the advent of Industry 4.0, businesses are moving at a fast pace towards automation, be it any type of industry. In 2013, researchers at oxford university did a study on the future of work. They concluded that almost one in every two jobs have a high risk of being automated by machines. Machine learning is responsible for this disruption. It is the most powerful branch of artificial intelligence. It allows machines to learn from data and mimic some of the things that humans can do.A research was conducted by the employees of Kaggle wherein an algorithm was to be created to take images of a human eye and diagnose an eye disease known as diabetic retinopathy. Here, the winning algorithm could match the diagnosis given by human ophthalmologists. Another study was conducted wherein an algorithm should be created to grade high school essays. Here too, the winning algorithm could match the grade given by human teachers.Thus, we can safely conclude that given the right data, machines can easily outperform human beings in tasks like these. A teacher might read 10,000 essays over a 40-year career; an ophthalmologist might see 50,000 eyes but a machine can read a million essays and see a million eyes within minutes.Thus, it is convenient to conclude that we have no chance of competing with machines on frequent, high volume tasks.\xa0Tasks where machines don’t workBut there are tasks where human beings have an upper hand, and that is, in novel tasks. Machines can’t handle things they haven’t seen many times before. The fundamental rule of machine learning is that it learns from large volumes of past data. But humans don’t; we have the ability of seemingly connecting disparate threads to solve problems we haven’t seen before.\xa0Percy Spencer was a physicist working on radar during world war 2 where he noticed that the magnetron was melting his chocolate bar. Here, he was able to connect his understanding of electromagnetic radiation with his knowledge of cooking in order to invent the microwave oven. Now this sort of cross pollination happens to each one of us several times in a day. Thus, machines cannot compete with us when it comes to tackling novel situations.\xa0Now as we all know that around 92% of talented professionals believe that soft skills such as human interactions and fostering relationships matter much more than hard skills in being successful in managing a workplace. Perhaps, these are the kind of tasks that machines can never compete with humans at.\xa0Also, creative tasks: the copy behind a marketing campaign needs to grab customers’ attention and will have to stand out of the crowd. Business strategy means finding gaps in the market and accordingly working on them. Since machines cannot outperform humans in novel tasks, it will be humans who would be creating these campaigns and strategies.\xa0Human contact would be essential in care-giving and educational-related work responsibilities, and technology would take a backseat. Health screenings and customer service face-to-face communication would advocate for human contact, with Artificial Intelligence playing a supporting role.\xa0So, what does this mean for the future of work? The future state of any single job lies in the answer to one single question: to what extent is the job reducible to tackling frequent high-volume tasks and to what extent does it involve tackling novel situations?Today machines diagnose diseases and grade exam papers, over the coming years they’re going to conduct audits, they’re going to read boilerplate from legal contracts. But does that mean we’re not going to be needing accountants and lawyers? Wrong. We’re still going to need them for complex tax structuring, for path breaking litigation. It will only get tougher to get these jobs as machine learning will shrink their ranks.\xa0Amazon has recruited more than 100,000 robots in its warehouses to help move goods and products around more effectively, and its warehouse workforce has expanded by more than 80,000 people. Humans pick and pack goods (Amazon has over 480,000,000 products on its “shelves”), while robots move orders throughout the enormous warehouses, therefore reducing “the amount of walking required of workers, making Amazon pickers more efficient and less exhausted.” Furthermore, because Amazon no longer requires aisle space for humans, the robots enable Amazon to pack shelves together like cars in rush-hour traffic.” More inventory under one roof offers better selection for customers, and a higher density of shelf space equals more inventory under one roof.Kodak Vs InstagramKodak, once an undisputed giant of the photography industry, had a 90% share in the USA market in 1976, and by 1984, they were employing 1,45,000 people. But in the year 2012, they had a net worth of negative $1 billion and they had to declare bankruptcy. Why? Because they failed to predict the importance of exponential trends when it comes to technology. On the other hand, Instagram, a digital photography company started in 2012 with 13 employees and later they were sold to Facebook for $1 billion. This is so ironic because Kodak pioneered digital photography and actually invented the first digital camera but unfortunately thought of it as a mere product and didn’t pay attention towards it and this created the problem.\xa0\xa0We live in an era of artificial intelligence (AI), which has given us tremendous computing power, storage space, and information access. We were given the spinning wheel in the first, electricity in the second, and computers in the third industrial revolution by the exponential growth of technology.Airbnb and its breakthrough idea!Airbnb, which is a giant start-up and is known for enabling homeowners to rent out their homes and couches to travellers, for example, “is now creating a new Artificial Intelligence system that will empower its designers and product engineers to literally take ideas from the drawing board and convert them into actual products almost instantly.” This might be a significant breakthrough whether you’re a designer, engineer, or other type of technologist.Differences that Automation brings onto the table:\xa0There are three key changes that automation can bring about at the macro level:\xa0Changes in capability demandGender imbalance in workforce redeploymentFirm reorganization.\xa0Sectors that might be in troubleArtificial Intelligence isn’t just a fad. Tractica, a market research firm, published a report in 2016 that predicted “annual global revenue for artificial intelligence products and services will expand from 643.7 million in 2016 to $36.8 billion by 2025, a 57-fold increase over that time span.” As a result, it is the IT industry’s fastest-growing segment of any size.”The reduction in need for people as a result of Artificial Intelligence and related technologies, which resulted in job layoffs, was a cause of fear. In India alone, job losses in the IT sector have reportedly reached 1,000 in the last year, owing to the integration of new and advanced technologies like artificial intelligence and machine learning.\xa0Most of the IT companies such as Infosys, Wipro, TCS, and Cognizant have reduced their employee base in India and are recruiting less, while engaging more personnel in the United States and investing heavily in “centres of innovation.” Artificial Intelligence and data science, which are currently the trending aspects that require fewer people and are primarily located abroad, aren’t helping the prospects of local employees. Another factor is that the computer industry is continuously growing and would develop to a size of two million workers. Unfortunately, it’s a drop in the bucket compared to what robots are doing to Information Technology’s less-skilled brothers.\xa0Large e-commerce sites that used to be operated by armies of people are now manned by 200 robots produced by GreyOrange, which is an Indian company based out in Gurgaon. These indefatigable robots lift and stack boxes 24 hours a day, with only a 30-minute break for recharging, and have cut employees by up to 80%. For efficiency, this is a victory but a disaster for job prospects.\xa0Concluding remarksInternal re-skilling and redeployment of staff is a critical requirement of the hour. Artificial intelligence has presented Indian policymakers with epistemological, scientific, and ethical issues. This requires us to abandon regular, linear, and non-disruptive mental patterns. The tale of artificial intelligence’s influence on individuals and their occupations will only be told over time. It is up to us to upskill ourselves and look for ways to stay current with the industry’s current trends and demands.\xa0So, will machines be able to take over many of our jobs? The answer is a resounding yes. However, for every job that is taken over by robots, there will be an equal number of positions available for people to do. Some of these human vocations will be artistic in nature. Others will necessitate humans honing superhuman cognitive abilities. Humans and machines can form symbiotic relationships, assisting each other in doing what they do best. In the future, people and machines may be able to collaborate and work together towards a common goal for any business they work for.\xa0Blackcoffer Insights 29:  Syed Basir Quadri and Sanchita Khattar, K J Somaiya Institute of Management'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[5]:


#10
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='Before the internet, information was in some ways restricted and more centralized. The only mediums of information were books, newspapers, and word of mouth, etc. But now with the advent of the internet and improvements to computer technology (Moore’s Law), information and data skyrocketed, and it has become this open-system, where information can be distributed to people without any kind of limits.SECURING YOUR DEVICES AND NETWORKSEncrypt your dataVarious publicly available tools have taken the rocket science out of encrypting (and decrypting) email and files. Data encryption isn’t just for technology geeks; modern tools make it possible for anyone to encrypt emails and other information. “Encryption used to be the sole province of geeks and mathematicians, but a lot has changed in recent years. In particular, various publicly available tools have taken the rocket science out of encrypting (and decrypting) email and files. GPG for Mail, for example, is an open-source plug-in for the Apple Mail program that makes it easy to encrypt, decrypt, sign and verify emails using the OpenPGP standard. And for protecting files, newer versions of Apple’s OS X operating system come with FileVault, a program that encrypts the hard drive of a computer. Those running Microsoft Windows have a similar program. This software will scramble your data, but won’t protect you from government authorities demanding your encryption key under the Regulation of Investigatory Powers Act (2000), which is why some aficionados recommend TrueCrypt, a program with some very interesting facilities, which might have been useful to David Miranda,” explains John Naughton in an article for The Guardian.Backup your dataOne of the most basic, yet often overlooked, data protection tips is backing up your data. Basically, this creates a duplicate copy of your data so that if a device is lost, stolen, or compromised, you don’t also lose your important information. As the U.S. Chamber of Commerce and insurance company Nationwide points out, “According to Nationwide, 68% of small businesses don’t have a disaster recovery plan. The problem with this is the longer it takes you to restore your data, the more money you’ll lose. Gartner found that this downtime can cost companies as much as $300,000 an hour.” The cloud provides a viable backup optionWhile you should use sound security practices when you’re making use of the cloud, it can provide an ideal solution for backing up your data. Since data is not stored on a local device, it’s easily accessible even when your hardware becomes compromised. “Cloud storage, where data is kept offsite by a provider, is a guarantee of adequate disaster recovery,” according to this post on TechRadar. Twitter: @techradarAnti-malware protection is a must.Scammers are sneaky: sometimes malware is cleverly disguised as an email from a friend or a useful website. Malware is a serious issue plaguing many computer users, and it’s known for cropping up in inconspicuous places, unbeknownst to users. Anti-malware protection is essential for laying a foundation of security for your devices. “Malware (short for malicious software) is software designed to infiltrate or damage a computer without your consent. Malware includes computer viruses, worms, trojan horses, spyware, scareware, and more. It can be present on websites and emails or hidden in downloadable files, photos, videos, freeware, or shareware. (However, it should be noted that most websites, shareware, or freeware applications do not come with malware.) The best way to avoid getting infected is to run a good anti-virus protection program, do periodic scans for spyware, avoid clicking on suspicious email links or websites. But scammers are sneaky: sometimes malware is cleverly disguised as an email from a friend or a useful website. Even the most cautious of web-surfers will likely pick up an infection at some point.,” explains Clark Howard. Twitter: @ClarkHowardFig: Protect your Computer from VirusesMake your old computers’ hard drives unreadable.Much information can be gleaned through old computing devices, but you can protect your personal data by making hard drives unreadable before disposing of them. “Make old computers’ hard drives unreadable. After you back up your data and transfer the files elsewhere, you should sanitize by disk shredding, magnetically cleaning the disk, or using software to wipe the disk clean. Destroy old computer disks and backup tapes,” according to the Florida Office of the Attorney General. Twitter: @AGPamBondiInstall operating system updates.Operating system updates are a gigantic pain for users; it’s the honest truth. But they’re a necessary evil, as these updates contain critical security patches that will protect your computer from recently discovered threats. Failing to install these updates means your computer is at risk. “No matter which operating system you use, it’s important that you update it regularly. Windows operating systems are typically updated at least monthly, typically on so-called ‘Patch Tuesday.’ Other operating systems may not be updated quite as frequently or on a regular schedule. It’s best to set your operating system to update automatically. The method for doing so will vary depending upon your particular operating system,” says PrivacyRights.org. Twitter: @PrivacyTodayAutomate your software updates.Many software programs will automatically connect and update to defend against known risks.In order to ensure that you’re downloading the latest security updates from operating systems and other software, enable automatic updates. “Many software programs will automatically connect and update to defend against known risks. Turn on automatic updates if that’s an available option,” suggests StaySafeOnline.org. Twitter: @StaySafeOnlineSecure your wireless network at your home or business.A valuable tip for both small business owners and individuals or families, it’s always recommended to secure your wireless network with a password. This prevents unauthorized individuals within proximity to hijack your wireless network. Even if they’re merely attempting to get free Wi-Fi access, you don’t want to inadvertently share private information with other people who are using your network without permission. “If you have a Wi-Fi network for your workplace, make sure it is secure, encrypted, and hidden. To hide your Wi-Fi network, set up your wireless access point or router so it does not broadcast the network name, known as the Service Set Identifier (SSID). Password protect access to the router,” says FCC.gov in an article offering data protection tips for small businesses. Twitter: @FCCTurn off your computer.When you’re finished using your computer or laptop, power it off. Leaving computing devices on, and most often, connected to the Internet, opens the door for rogue attacks. “Leaving your computer connected to the Internet when it’s not in use gives scammers 24/7 access to install malware and commit cybercrimes. To be safe, turn off your computer when it’s not in use,” suggests CSID, a division of Experian. Twitter: @ExperianPS_NAFig: To Avoid from Hacking turn off your ComputerUse a firewall.Firewalls assist in blocking dangerous programs, viruses, or spyware before they infiltrate your system.”Firewalls assist in blocking dangerous programs, viruses, or spyware before they infiltrate your system. Various software companies offer firewall protection, but hardware-based firewalls, like those frequently built into network routers, provide a better level of security,” says Geek Squad. Twitter: @GeekSquadPractice the Principle of Least Privilege (PoLP).Indiana University Information Technology recommends following the Principle of Least Privilege (PoLP): “Do not log into a computer with administrator rights unless you must do so to perform specific tasks. Running your computer as an administrator (or as a Power User in Windows) leaves your computer vulnerable to security risks and exploits. Simply visiting an unfamiliar Internet site with these high-privilege accounts can cause extreme damage to your computer, such as reformatting your hard drive, deleting all your files, and creating a new user account with administrative access. When you do need to perform tasks as an administrator, always follow security procedures.” Twitter: @IndianaUnivUse “passphrases” rather than “passwords.”What’s the difference? “…we recommend you use passphrases–a series of random words or a sentence. The more characters your passphrase has, the stronger it is.\xa0 The advantage is these are much easier to remember and type, but still hard for cyber attackers to hack.” explains SANS. Twitter: @SANSAwarenessEncrypt data on your USB drives and SIM cards.Encrypt your SIM card in case your phone is ever stolen, or take it out if you are selling your old cell phone. Encrypting your data on your removable storage devices can make it more difficult (albeit not impossible) for criminals to interpret your personal data should your device become lost or stolen. USB drives and SIM cards are excellent examples of removable storage devices that can simply be plugged into another device, enabling the user to access all the data stored on it. Unless, of course, it’s encrypted. “Your USB drive could easily be stolen and put into another computer, where they can steal all of your files and even install malware or viruses onto your flash drive that will infect any computer it is plugged in to. Encrypt your SIM card in case your phone is ever stolen, or take it out if you are selling your old cell phone,” according to Mike Juba in an article on Business2Community. Twitter: @EZSolutionCorpDon’t store passwords with your laptop or mobile device.A Post-It note stuck to the outside of your laptop or tablet is “akin to leaving your keys in your car,” says The Ohio State University’s Office of the Chief Information Officer. Likewise, you shouldn’t leave your laptop in your car. It’s a magnet for identity thieves. Twitter: @OhioStateFig: Media SharingDisable file and media sharing if you don’t need it.If you don’t really need your files to be visible to other machines, disable file and media sharing completely. If you have a home wireless network with multiple devices connected, you might find it convenient to share files between machines. However, there’s no reason to make files publicly available if it’s not necessary. “Make sure that you share some of your folders only on the home network. If you don’t really need your files to be visible to other machines, disable file and media sharing completely,” says Kaspersky. Twitter: @kasperskyCreate encrypted volumes for portable, private data files.HowToGeek offers a series of articles with tips, tricks, and tools for encrypting files or sets of files using various programs and tools. This article covers a method for creating an encrypted volume to easily transport private, sensitive data for access on multiple computers. Twitter: @howtogeeksiteOverwrite deleted files.Deleting your information on a computing device rarely means it’s truly deleted permanently. Often, this data still exists on disk and can be recovered by someone who knows what they’re doing (such as, say, a savvy criminal determined to find your personal information). The only way to really ensure that your old data is gone forever is to overwrite it. Luckily, there are tools to streamline this process. PCWorld covers a tool and process for overwriting old data on Windows operating systems. Twitter: @pcworldDon’t forget to delete old files from cloud backups.If you back up your files to the cloud, remember that even though you delete them on your computer or mobile device, they’re still stored in your cloud account. If you’re diligent about backing up your data and use a secure cloud storage service to do so, you’re headed in the right direction. That said, cloud backups, and any data backups really, create an added step when it comes to deleting old information. Don’t forget to delete files from your backup services in addition to those you remove (or overwrite) on your local devices. “If you back up your files to the cloud, remember that even though you delete them on your computer or mobile device, they’re still stored in your cloud account. To completely delete the file, you’ll also need to remove it from your backup cloud account,” says re/code. Twitter: @RecodeBlackcoffer Insights 28: Monica V, SNS COLLEGE OF TECHNOLOGY'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[6]:


#11
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='We all hear day in and day out that we amidst a technological revolution. But do we know what this really means?Before we understand how its going to impact us, let us first discuss what these terms really mean.A technological revolution simply means that we are in a period where better and newer technologies replace the others to get the job done faster and better. We are in an era with rapid innovations where machines are being compared to humans.Fig: Technological RevolutionSo, then what is Machine Learning?Machine Learning is basically the application of artificial intelligence into electronic systems to enable them to learn and enhance themselves without being programmed by humans. It is the evolution and development of computer programs that can access data and then use it to advance themselves. Whether you know it or not, you use machine learning-powered applications daily.Now, what is Artificial intelligence?At its simplest form, artificial intelligence is a field, which combines computer science and robust datasets, to enable problem-solving.\xa0In simple words, Artificial Intelligence is the technology that facilitates these machines to perform human like behaviour.Just like every other industry, machine learning is playing its role in the finance and banking industry too. In most cases where a human would perform the same task by performing the same calculations or following the same process can be taught to the machine which can now perform it by itself.Let us discuss a few examples of the applications that we might have come in our day to day running which are a result of machine learning in this industry:Portfolio ManagementRisk UnderwritingAlgorithmic TradingFraud DetectionProcess AutomationCustomer onboardingCustomer churnDecision MakingProcess AutomationPortfolio ManagementIn earlier days, an investor would need to consult a financial advisor to understand his/her risk appetite and advise accordingly. Today, using machine learning algorithms there exists the concept of a “Robo-Advisor” that requires any user to give certain inputs about their financial status and goals and calculates their risk tolerance and constructs and idle portfolio allocation for them. Young users today find this extremely useful rather than physically visiting an advisor and paying a fee for doing so.Risk UnderwritingUnderwriting is one of the core functions for most financial institutions especially banks and insurance companies where they are required to underwrite the risk of the customers before loaning out money or insurance policies. These underwriting activities are based on trends and thumb rules industrywide. The same has been introduced through machine learning which is able to underwrite risks today on a larger and more accurate scale.Algorithmic TradingMachine learning is a mathematical model that tracks market information, analyses massive data sources and study market conditions simultaneously to detect patterns which can be used for trading. This is humanly impossible to do in a fraction of time. Algorithmic systems can make millions of trades daily, often known as “high-frequency trading”. It is highly believed that deep learning is playing its role in calibrating real-time trading decisions.Fraud DetectionWith the increase in use and dependency on computers for financial transactions came the data security risk. There is an ample amount of valuable data stored online available to create potential risk. Machine learning thus helped in fraud detection by detecting anomalies in transactions and flagging them for scrutiny based on the risk factors defined by the institutions. Fraud identification in insurance claims, credit card payments, identity theft, account theft, are all areas in fraud detection that machine learning can help in.Process AutomationProcess automation is one of the most common applications of machine learning in finance. The technology has helped in replacing manual work, automate repetitive tasks to avoid redundancy, and as a result, increase productivity. Machine learning has benefitted these organizations to optimize costs, improve customer experiences, and scale up their services. Some examples of financial and banking firms using process automation are the use of chatbots, automated calls, paperwork automation, and gamified employee training.Customer onboardingIn this highly competitive industry, customer acquisition and the customer onboarding process is highly relevant in building a good customer relationship. At any stage, during the onboarding, a slight inconvenience or delay can act as a barrier. Machine learning-enabled complete automation in this process for these financial and banking institutions. Today, from opening an account, filing for any application can be completed within a few minutes with utmost ease. With AI, customers’ behavioral patterns have been studied to improvise and make the whole process efficient and user-friendly.Customer churnWith the multitude of offerings and availability of a plethora of options, customer stickiness is a big problem faced by financial firms. Customer churn forecasting is one of the best big data use cases. It helps in detecting customers who cancel their subscription and analyses the same to tailor products as per customer needs. Video streaming application, Netflix’s subscribers worldwide has continued to grow to reach 167 million through using machine learning analytics on their customer database.Decision MakingFinancial and banking institutions function on facilitating investments made by their customers. Organizations are constantly in search of customers from whom they can get more revenue. This is now possible through performing machine learning analysis on both structured and unstructured data which helps them make more informed decisions. It also analyses data from the website and mobile application to construct effective marketing campaigns for the targeted customers.Future of Machine Learning\xa0in FinanceFinancial monitoring, security analysis, prevention of money laundering, network security, investment predictions, personalization of customer service everything comes under the realm of the applications of machine learning in the financial and banking industry. Yet, this is just the tip of the iceberg, there is a lot more that is going to change in the future. It is now visibly imperative that while AI is beginning to create a wave of transformation across these industries and adapting to these changes s important for one’s survival. With smart technology applied everywhere, all financial firms are bound to turn into FinTech’s to stay relevant to the “silver tech generation” consisting of millennials and the GenZs.Final thoughtsThe financial services industry has entered the space of artificial intelligence and machine learning, and the pace is not surprising knowing the positive changes it has brought. Machine learning has the most use cases in finance than any other industry because of the available computer power and new machine learning tools. The greatest applications include simplifying customer engagement and accurate sales forecasting. It is only making this industry better and more efficient with each new adaptation. Machine learning algorithms have the capability to deal with a lot more than human capacity along with eliminating human error. As even the algorithms are constantly learning and innovating, they can serve as a bridge to a completely flawless automated financial system in the future. Nonetheless, the challenges of high cost and lack of resources that come along play a significant role in how early these firms can adopt these technologies. But even then, the future seems bright as the industry has enough adopters and prospects ready to explore.Blackcoffer Insights 28: Tanisha Gupta, XLRI'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[7]:


#12
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='It’s the year 2060. An automaton in a Research Laboratory says to a Scientist, “Warning! Error Occurred Reformatting Hard Disk Now!” The scientist panics. Automaton says again,” Ha! Ha! Just Kidding! “.Funny Right. Before some of you say that this joke isn’t realistic, “How can an Automaton tell you a joke?” But what if I tell you in 2017, “Sofia” the robot made a joke on the show Good Morning Britain! Who thought computers could tell us a joke? Hard to believe? Well, the idea of giving computers human-enjoy thinking has now become a reality. Thanks to the technological advancement in AI in the last decade.Before diving deep into how AI can impact the future of work, let’s begin with the simple question: what’s AI? Artificial Intelligence provides machines the power to think from data. The machine uses the patterns and trends found in data and makes its decision, but cannot create thought beyond these patterns and trends.Fig: Impact of robots on employmentsWith the rise of AI, humans are divided into one question. Are machines human’s friend or foe? Tech executives and politicians on conference stages, campaign rallies, and even science fiction Hollywood movies like Carbon Black, Westworld, Minority Report, and Ex Machina have given their take on this question. Some believe AI will help us solve problems while others believe that the rise of AI will result in destruction and maybe the end of the world, we all know.Stephen Hawking made it no secret of his concern about the rise of superhuman AI that eventually would escape earth to a new planet. No, this isn’t a plot of Black Mirror. Right now, Superhumans may not be a reality, but AI is.“Homo Deus”, the emergence of the new Digital God using AI. God must also worry, as AI might take his job.Here’s some Career Advice, have you thought about being a Robot? The fear that AI would automate all jobs in the future eventually leaving all humans jobless has been daunting for many workers today. Statistics show that nearly 37% of workers worry about losing their jobs to robots. While another thought that many people believe is that though the rise of AI with result in automating most of the jobs in the future, however, it also will create millions of new job opportunities.AI is already replacing most manual and repetitive tasks. For example, buying a metro ticket or a movie ticket is now almost a human-less interaction. Each year the number of industrial robot jobs increases by 14 %. At this rate, it’s predicted that the 20 million jobs in the manufacturing industry will be replaced by robots due to automation.The coronavirus pandemic and recession have boosted the demand for automation. The Robotic Process Automation (RPA) Software industry has experienced an increase of 19.53% in the year 2021. Coronavirus pandemic has increased interest in technology that reduces human contact as minimal for making workplaces safe.Our workplaces will look much different in the next five to ten years. AI will help humans in simplifying repetitive processes. The two most important catalysts for the future of work are the two D’s- Digitization and Datafication. Digitalization is converting data to digital formats (computer-readable). For example, text to Html, analog video to YouTube video. Digitization helps in increasing data exponentially. Datafication is quantifying human life to data and improving the data-driven business model. By 2025, it is forecasted that the digital transformation space will build in a $3,294 billion industry!One thing is clear, no data, no future of work. What we find is that the future of data and the future of work will go hand in hand. The total volume of data in the datasphere that is created, captured, copied, and consumed in the world is predicted to reach 175 zettabytes by 2025. To give you a much better picture for understanding, if we represent the digital universe as stacks of tablets, there would be 27.25 stacks from earth to the moon.It’s time to prepare for the data-dominated future as Industry 4.0/Fourth Industrial Revolution has begun. So, let’s see how artificial intelligence will affect the following fields:Human Resource: Nowadays, recruiters use AI-powered tools for hiring workers. Using these tools, recruiters get insights into a candidate’s skills, personality and even check whether the candidate is fit for the organization. For example, the company AllyO first identifies high-potential candidates through assessment and smart screening, and then automatically schedules interviews using AI. HR departments at large companies receive hundreds of resumes for a job opening. Entry-level roles focusing on screening and scheduling will be automated. AI will automate specific HR jobs, not HR roles. A Deloitte study found that AI has already eliminated 800,000 low-skilled jobs in the UK, but 3.5 million new jobs were also created. Roles that focus on complex decisions like resolving disputes within a department will continue to be a very human endeavor.Finance and Accounting: In 2015, a report from Accenture named “Finance 2020: death by digital” predicted that 40 percent of transactional accounting work would be automated by 2020. Has technology replaced the human factor? Well, AI has created new jobs involving managing the AI system and using the information to create insights. For example, accounting software has already automated bookkeeping tasks that used to be done by humans, but that’s only opened the door for former bookkeepers to learn skills needed to run and manage the software for employers and clients. Advisors are another crucial role of the accounting and finance team. Using the information gained from transactions in books, the team creates insights to improve business strategy. Owing to automation, the team spends more time analyzing numbers.Marketing and Sales: Marketing automation has helped companies strategize the proper utilization of the company’s resources, managing time, and achieving budget targets. Marketing automation has helped to draw conclusions at a scale no marketer ever would. In this process, marketers and machines both excel in different parts. Marketers using AI tools drive more conversions in less time. Human Intelligence with technology can help identify the right customers to talk to and at the right time. Modern Marketers understand the insights from any marketing campaign and create it into effective messaging.Engineering: Technology is changing in a blink of an eye. The technologies used five years ago in the industry have become obsolete today. Engineers will have to keep up with the technological advancements and keep upgrading their skills to stay relevant in the industry. Learning to work alongside machines and designing work such that interaction better humans and machines are better are going to be important skills for engineers in the future.Fig: Robots takes place the job role of man.In the 18th and 19th centuries, the rise of the industrial revolution centuries led to millions of people losing their jobs because of scientific advancements. But that also ended in creating millions of other jobs. Statisticians have said, when automation destroys jobs, people find new ones. Thus, AI holds a more optimistic picture for the future.In the future, AI is not going to replace humans, rather make jobs more humane. AI will disrupt millions of middle and entry-level jobs in the next few years but will also create millions of additional jobs and help to boost economies.Blackcoffer Insights 28: DAVANG SIKAND, Manipal University Jaipur'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[8]:


#13
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='It’s the year 2060. An automaton in a Research Laboratory says to a Scientist, “Warning! Error Occurred Reformatting Hard Disk Now!” The scientist panics. Automaton says again,” Ha! Ha! Just Kidding! “.Funny Right. Before some of you say that this joke isn’t realistic, “How can an Automaton tell you a joke?” But what if I tell you in 2017, “Sofia” the robot made a joke on the show Good Morning Britain! Who thought computers could tell us a joke? Hard to believe? Well, the idea of giving computers human-enjoy thinking has now become a reality. Thanks to the technological advancement in AI in the last decade.Before diving deep into how AI can impact the future of work, let’s begin with the simple question: what’s AI? Artificial Intelligence provides machines the power to think from data. The machine uses the patterns and trends found in data and makes its decision, but cannot create thought beyond these patterns and trends.Fig: Impact of robots on employmentsWith the rise of AI, humans are divided into one question. Are machines human’s friend or foe? Tech executives and politicians on conference stages, campaign rallies, and even science fiction Hollywood movies like Carbon Black, Westworld, Minority Report, and Ex Machina have given their take on this question. Some believe AI will help us solve problems while others believe that the rise of AI will result in destruction and maybe the end of the world, we all know.Stephen Hawking made it no secret of his concern about the rise of superhuman AI that eventually would escape earth to a new planet. No, this isn’t a plot of Black Mirror. Right now, Superhumans may not be a reality, but AI is.“Homo Deus”, the emergence of the new Digital God using AI. God must also worry, as AI might take his job.Here’s some Career Advice, have you thought about being a Robot? The fear that AI would automate all jobs in the future eventually leaving all humans jobless has been daunting for many workers today. Statistics show that nearly 37% of workers worry about losing their jobs to robots. While another thought that many people believe is that though the rise of AI with result in automating most of the jobs in the future, however, it also will create millions of new job opportunities.AI is already replacing most manual and repetitive tasks. For example, buying a metro ticket or a movie ticket is now almost a human-less interaction. Each year the number of industrial robot jobs increases by 14 %. At this rate, it’s predicted that the 20 million jobs in the manufacturing industry will be replaced by robots due to automation.The coronavirus pandemic and recession have boosted the demand for automation. The Robotic Process Automation (RPA) Software industry has experienced an increase of 19.53% in the year 2021. Coronavirus pandemic has increased interest in technology that reduces human contact as minimal for making workplaces safe.Our workplaces will look much different in the next five to ten years. AI will help humans in simplifying repetitive processes. The two most important catalysts for the future of work are the two D’s- Digitization and Datafication. Digitalization is converting data to digital formats (computer-readable). For example, text to Html, analog video to YouTube video. Digitization helps in increasing data exponentially. Datafication is quantifying human life to data and improving the data-driven business model. By 2025, it is forecasted that the digital transformation space will build in a $3,294 billion industry!One thing is clear, no data, no future of work. What we find is that the future of data and the future of work will go hand in hand. The total volume of data in the datasphere that is created, captured, copied, and consumed in the world is predicted to reach 175 zettabytes by 2025. To give you a much better picture for understanding, if we represent the digital universe as stacks of tablets, there would be 27.25 stacks from earth to the moon.It’s time to prepare for the data-dominated future as Industry 4.0/Fourth Industrial Revolution has begun. So, let’s see how artificial intelligence will affect the following fields:Human Resource: Nowadays, recruiters use AI-powered tools for hiring workers. Using these tools, recruiters get insights into a candidate’s skills, personality and even check whether the candidate is fit for the organization. For example, the company AllyO first identifies high-potential candidates through assessment and smart screening, and then automatically schedules interviews using AI. HR departments at large companies receive hundreds of resumes for a job opening. Entry-level roles focusing on screening and scheduling will be automated. AI will automate specific HR jobs, not HR roles. A Deloitte study found that AI has already eliminated 800,000 low-skilled jobs in the UK, but 3.5 million new jobs were also created. Roles that focus on complex decisions like resolving disputes within a department will continue to be a very human endeavor.Finance and Accounting: In 2015, a report from Accenture named “Finance 2020: death by digital” predicted that 40 percent of transactional accounting work would be automated by 2020. Has technology replaced the human factor? Well, AI has created new jobs involving managing the AI system and using the information to create insights. For example, accounting software has already automated bookkeeping tasks that used to be done by humans, but that’s only opened the door for former bookkeepers to learn skills needed to run and manage the software for employers and clients. Advisors are another crucial role of the accounting and finance team. Using the information gained from transactions in books, the team creates insights to improve business strategy. Owing to automation, the team spends more time analyzing numbers.Marketing and Sales: Marketing automation has helped companies strategize the proper utilization of the company’s resources, managing time, and achieving budget targets. Marketing automation has helped to draw conclusions at a scale no marketer ever would. In this process, marketers and machines both excel in different parts. Marketers using AI tools drive more conversions in less time. Human Intelligence with technology can help identify the right customers to talk to and at the right time. Modern Marketers understand the insights from any marketing campaign and create it into effective messaging.Engineering: Technology is changing in a blink of an eye. The technologies used five years ago in the industry have become obsolete today. Engineers will have to keep up with the technological advancements and keep upgrading their skills to stay relevant in the industry. Learning to work alongside machines and designing work such that interaction better humans and machines are better are going to be important skills for engineers in the future.Fig: Robots takes place the job role of man.In the 18th and 19th centuries, the rise of the industrial revolution centuries led to millions of people losing their jobs because of scientific advancements. But that also ended in creating millions of other jobs. Statisticians have said, when automation destroys jobs, people find new ones. Thus, AI holds a more optimistic picture for the future.In the future, AI is not going to replace humans, rather make jobs more humane. AI will disrupt millions of middle and entry-level jobs in the next few years but will also create millions of additional jobs and help to boost economies.Blackcoffer Insights 28: DAVANG SIKAND, Manipal University Jaipur'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[9]:


#14
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='Through AI tools like natural language processing, Alexa and google assistant has led the retail industry in its rise towards conversational commerce. As if a customer was interacting with a clerk in a retail store, conversational commerce makes it possible for users to engage with software to research, purchase, or get customer assistance with products and services across a wide range of industries. With Alexa, for example, users can ask any Alexa-enabled device to add an item to an Amazon shopping cart, set a purchasing reminder when a product is running low, or carry out a complete purchase without having to access a shopping cart. The result is a seamless conversational experience that enables consumers to carry out transactions as quickly as it takes to speak a sentence.Through AI tools like natural language processing, Alexa has led the retail industry in its rise towards conversational commerce. As if a customer was interacting with a clerk in a retail store, conversational commerce makes it possible for users to engage with software to research, purchase, or get customer assistance with products and services across a wide range of industries.With the advent of personalized products and on-call delivery, customers have come to expect a new standard experience: fast, easy, accurate, and personalized. Accomplishing this without sacrificing your workday can be a challenge, since the data processing required to meet these needs is immense. Luckily, virtual agents (VAs), powered by conversational AI, can utilize this information faster and more accurately than humans, finding insights and automating communication to deliver an enriched customer experience. If you invest based on these improvements, you’ll find that implementing these tools delivers a powerful competitive advantage. AI has helped in automobile, education, retail and commerce, finance and banking and healthcare.Voice AI has powered the wheels of conversational e-commerce, which has impacted the way the customer communicates with the brand in multiple industries. Brands generally build a campaign to emotionally connect with customers, for long-term growth. With Voice, brand campaigns need to be short and ones that can lead to immediate buying. Conversational e-commerce is still in its nascent stage and it is expected to grow manifold in the coming years. The future of shopping is going to Voice AI and marketers have to get on the bandwagon fast to increase their brand value and visibility. Targeting will have to be highly personalized for success.Despite its narrow focus, conversation AI is an extremely lucrative technology for enterprises, helping businesses more profitable. While an AI chatbot is the most popular form of conversational AI, there are still many other use cases across the enterprise. While an exclusively chat- or voice-based shopping experience for all scenarios may never completely replace the in-person experience, conversational commerce will continue to grow as an added method of convenient and efficient communication. As users continue to become more accustomed to engaging with chatbots and voice-driven interfaces, expect more innovations in the space as brands continue to develop their unique conversation-based solutions.Blackcoffer Insights 28: Samyak Jain'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[10]:


#15
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='AI experts believe it’s going to be one of the main drivers of the fourth Industrial Revolution and that it has the potential to not just transform the tech sectors and going to open a new chapter of the society of the world that people try to understand themselves better rather than the outside world with AI because people who are naysayer and kind of try to drum up these doomsday scenarios are pretty irresponsible. After all, In the next, five to ten years AI is going to deliver so many improvements and the quality of our lives it is a renaissance, a golden age of machine- learning and artificial intelligence that was the realm of science fiction for the last several decades. AI is probably the most important thing humanities that have ever worked which is more profound than any work with technology, as it is important to harness the benefits and while minimizing the downside is focusing on autonomous systems like self-driving cars seen as the mother of all AI projects and has made applications like self-driving technology viable for the first time, three things happen at the same time number one data collection and data processing became easier because of better technologies right um you need data to fuel AI training and that’s been one of the big drivers the second thing that has happened is that computer processing has become faster that’s like the engine so no matter how much fuel you have if you don’t have that engine and processing the data on a timeframe that’s reasonable was just not possible and the third thing that’s happened is that new algorithms have been developed which has made AI much more powerful so #technology has been changing and developing at a pace that’s much faster than ever before and we have not been used to this rapid pace of change which means that we have not been used to thinking about how it’s going to impact our immediate future. The most important factor responsible for the growth of AI is Google and its AI what Google’s done is given all of us the power to get the relevant information we want at our fingertips this has created a shift in how things are bought but it didn’t happen overnight this started in 2004 but the major change only happens to start 2012 onwards Google’s taken away about 65% of sales people’s jobs that were primarily order takers and the ones that are remaining are likely to be gone over the next decade.In present several AI projects are helping in diagnosing diseases better match up drugs with people depending on what they’re sick they can get treated better so it’s going to help a whole lot of people get treated and get better #healthcare than would have had access to it before if you look at self-driving cars they’re going to be safer than people driving cars and the value that machine learning is providing is actually happening beneath the surface and it is things like improved search results improved product recommendations for customers improve forecasting for inventory management and literally hundreds of other things including speech-recognition or image-recognition that the performance levels are phenomenal  or drug discovery as these biological systems are very complicated because vaccines for TB and HIV developing that’s notably enabled by this rich data advanced in biology and machine learning and recent invention in which is an application we just launched for anybody with visual impairment ass it uses the latest cutting-edge computer vision technology to give anyone the ability to see, so anyone who has dyslexia can now use AI to be able to read better and with the  latest release of Windows 10 has this capability called IJ’s which enable the eye muscle that the gaze can help to type. Like the two sides of the coin, there are negative impacts of AI as well  Bill Gates Ellen Musk also tech giants in a way their views are pessimistic, to say the least, they warned against the potential of AI to replace humans in the workplace and Ella masks even went as far as to claim that AI is the biggest existential threat to mankind. because of the loss of a job, when you think about a job or a career choice if a majority of the tasks that comprise that career choice is likely to be these vulnerable tasks then that is a career at risk in the future so what are the tasks that AI will find? hard to do anything unpredictable anything that requires skills like creative thinking or empathy or interpersonal skills but it’s important to understand tomorrow whether Google is there or not, artificial intelligence is going to progress you know technology has just nature it’s that it’s going to evolve as  technology and in particular AI can, in fact, bring more empowerment more inclusiveness and at the same time it is important to be clear-eyed about displacement and unintended consequences like any other technology and work both skills so that people can find the jobs of the future create new jobs also the policy decisions that help people as they go through this change people already unhappy because of machine learning artificial intelligence as they think  if they’re not innovative enough or not creative enough your job will be taking away by a lot of machines AI for business going to affect the future of work specifically there are jobs that are at more risk of being taken over by AI and automation there is very wide dissonance on this, there are different reports that have been shared by  Oxford study that says 47% of US jobs are at risk of automation over the next few years meanwhile the general population and workers think differently a recent study conducted by college actually identifies that 97% of workers believe that most jobs will be automated but not their own this suggests that the general public needs to be educated on which jobs are  susceptible to this risk which are not and businesses need to be aware of the forthcoming skills gap of course not all jobs are equal the Oxford study that highlights this they examined 700 participants and found the generalist occupations that require creative knowledge or innovation are at least risk the same is true for occupations in education healthcare media and arts jobs on the flip side jobs like telemarketers junior lawyers accountants are at most risk in short there is a simple rule of thumb if your job is in some way predictable or routine the risk of automation is much higher if a job doesn’t require innovation or creativity then the return on investment for companies is higher on machines than real-time employees machines are faster can’t be distracted and can work 24/7 this is actually good for creative marketers because AI and automation can serve to augment their jobs rather than substituting them as impact of emerging technologies on the creative economy they stated that artificial intelligence is changing  creative content from beginning to end by 2030 AI will be able to write high school essays code in Python composed top 40s chart songs and make creative videos but all of these advancements also come with risks and costs take a look at this report by the global Commission on the future of work in the absence of effective transition policies many people will have to accept lower skilled and lower paying jobs high-skilled workers are taking less cognitively demanding jobs displacing less educated workers and this is already happening also technological dividends are being unevenly distributed among firms a very limited amount of companies tend to dominate when it comes to big data just think about Google and Facebook today they alone are responsible for 70% of the referral marketing traffic and  receive more than 50% of total global advertising budget so the question is in businesses workers and social institutions go into the same direction if companies and public policy leaders can understand the evolving landscape they can help the workforce anticipate the upcoming challenges technology and the demographic changes are leading to a smaller workforce compared to the previous generation and a workforce that has to pursue many careers during their time of work we need to provide workers with an environment where they can continuously upskill and grow governments will have to re-evaluate the educational system we will have to continuously learn and grow and companies will have to redesign their structure and their culture around technology just like during the Industrial Revolution we are heading into a new age and the great transformation that we’re about to see by 2022 it is estimated that 20 to 25 percent of the labor force will be displaced within 10 to 20 years however this is also an opportunity for people to get ahead for which different ways have to be find to attract and retain highly skilled workers and allow them the time to up skill themselves even during work hours and it is a  good way  to develop a learning community to benefit from each other and also to use technology to supplement goal tracking and  efforts instead of as a distraction in short what we are doing is  to bridge the dissonance and it is imperative to build a  map of how AI and automation will affect  industry and  company if this is an economic imperative how do people feel about committing itself to a lifelong approach to knowledge as  these risks are important but it is important to do things like from being upfront to have ethical charters like AI safety and to be very transparent and open and how we perceive progress there and figure out global frameworks by which we can engage just like Paris agreement and climate change by using  such forums bring people together as they engage on the hard questions and it will emerge answers and on the question of whether AI is a threat or not, artificial intelligence is not  a threat because there is a rare case where people need to be proactive in regulation instead of reactive because I think by the time we are reactive in AI regulation it’s too late right now we have machine learning algorithms that can solve an incredibly complex problem beyond any human intelligence  as they are mere machines that can be given enormous data set and they come up with brilliant correlations and insights but they’re not going threaten the human population anytime soon because fish intelligent isn’t terrible but human being a smart enough to learn that skills at least to have a complete toolbox to be prepared volatility of the future adaptability.Blackcoffer Insights 28: Mihir Bhatt, Delhi University ( SGTB KHALSA College)'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[11]:


#16
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text=' Ever wondered how you get notified of the products or services you want or you have been looking for for a long time. Now how does this happen? Let me start with the simple and the most known fact- Marketing,Marketing is a common term which everyone knows and is aware of. Marketing is the action of promoting products and services, including market research and advertising.Traditional Marketing was working fine for all these years. So why there was a need for online marketing?The story begins from the internet era, The online presence of customers-There are 4.72 billion people are on internet users in the world today and the number is increasing day by day. So if the companies want to create awareness about their products and services their is a huge audience present online.There are a lot of benefits of online marketing like a large audience, benefits of targeting on the basis of demographics, location, age, gender, and many more. Because of this diversity, almost every type of company can use the Internet to reach any audience. All would find something of their liking.Let’s take a look at what does online marketing involves.Types of Online marketingSocial Media AdvertisingThese days, almost everybody is on social media. The majority of people use Facebook, Twitter, Instagram, and other social media platforms to communicate with their friends and relatives. Some people have created companies solely based on their social media activity.You can, however, promote your Knowledge Commerce products through social media. Whether or not you advertise on these sites.The best social media networks for advertising-Facebook advertisingInstagram advertisingTwitter advertisingPinterest advertisingLinkedIn advertisingSnapchat advertisingContent MarketingContent marketing is a strategic marketing strategy that focuses on producing and delivering useful, appropriate, and reliable content in order to attract and maintain a specific audience — and, eventually, to drive profitable consumer action.In simple words, content marketing is a marketing strategy that producers and delivers relevant and reliable content to attract potential clients and to also retain existing clients.PPC/Search advertisingAbout half of all website traffic originates from a search engine. People who use searches are often high-intent buyers. This indicates that they are searching for a particular item. They’re all set to buy the products and services.The majority of online marketing is focused on pay-per-click (PPC). However, when you hear the word “PPC,” it refers to search ads.Pay-per-click (PPC) advertising on search engines, social media sites, and other online venues can be extremely successful.So the next time when you search for a product and services and you find similar ads on the internet this is a kind of internet marketing.E-mail marketingEmail marketing is the highly successful digital marketing technique of sending emails to prospects and consumers. It allows communicating directly with the present, former, and potential customers. businesses will inform the audience about new products as they become aware of them.Banner advertisingBanner ads are rectangular or square advertisements that appear above, in the sidebar, or below the content on websites. Usually, a banner ad leads to a sales or landing page. You will get great results by running banner ads on websites that attract members of the target audience.Affiliate MarketingAn affiliate marketer, like a car salesperson, only gets paid when someone buys the stuff. You may not have to pay if there are no purchases.Affiliates are free to sell your goods anywhere they choose (as long as the material follows the terms of service of the website). It’s a fantastic way to reach new markets.Anyone can start their affiliate marketing career by registering on the different affiliate websites.Influencer MarketingInfluencer marketing is a new trend for online marketing. They have a strong following, can inspire people to buy your goods, and are loved by their viewers.The type of online marketing that will work best for the company will be determined by many factors, including the nature of your industry, the tastes and demographics of the target market, and budget. Market analysis will guide to the best strategy or combination of strategies for your offerings, and comprehensive performance metrics will show you which are the most effective.Advantages of Online MarketingOnline advertising provides a large client base for a company’s services or products. All kinds of companies, from multinationals to small and medium enterprises, have access to millions of potential customers. The higher the number of users who visit your website, the more sales you can make.One can advertise their company 24 hours a day, seven days a week, through online marketing campaigns. You just won’t have to think about employee pay or shop hours. Furthermore, time differences in different parts of the world will have no impact on your campaigns. In today’s ads, social media is important. This is due to the fact that customers read comments and feedback left by other customers on the internet. Businesses can easily integrate social media tools into their advertising strategies and benefit from consumers who use social media extensively.In an online marketing process, consumers can be demographically targeted even more effectively than in an offline process. Organizations will enhance their targeting over time, have a better understanding of their consumer base, and generate exclusive deals that are only shown to certain demographics when combined with the improved analytics.Online marketing is a rapidly expanding industry that benefits companies in a variety of ways. The number of people who purchase goods and services online is on the rise. As a result, an increasing number of businesses around the world are turning to internet marketing to communicate with consumers and advertise their goods and services. '
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[12]:


#17
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='Evolution of Advertising Industry over the years.Advertising can be described as a type of specific content broadcasting to a larger audience; the form can take several different forms, and the intended message can differ from genre to genre. The target for each medium could be different. Advertised content could be in print, radio, TV, or digital formats.We’ll look at how the advertising market has changed over the last ten decade.Advertising is a form of communication that aims to persuade a target audience. Typical advertising messages endorse programs, goods, concepts, people, and companies. First, conventional types of advertising were used to carry out advertising. Let’s take a look at the traditional forms of advertising.Advertising can include in-flight advertising, street furniture, passenger displays, billboards, skywriting, posters, wall paintings, banners, taxi cabs, passenger screens, television, and newspaper advertisements.Press advertising-Other types of advertising include press advertisements in magazines and newspapers. Advertising in the classified section of a newspaper is an example of press advertising. A billboard or digital screen placed on a moving vehicle is often referred to as a mobile billboard.Guerrilla advertising-When a brand or product is used in a large entertainment venue, it is known as convert ads or guerrilla advertising. When a soft drink, a watch, or a pair of sneakers is seen or mentioned in a common film, this is an example of this.In-store advertising-Ad in supermarket videos, aisles, and on the inside of shopping carts is referred to as in-store advertising.Consumers are influenced by celebrity advertisements because of the power of wealth, fame, and popularity. However, if a celebrity falls out of favor, the use of that celebrity may be detrimental to a company.Non-commercial ads-Religious organizations, political parties, political candidates, and special interest groups are examples of noncommercial ads.These were the conventional forms of advertisement, but as the internet and technology progressed, the advertising industry began to play a role in helping brands establish a digital presence and advertising their products in a new way.The advertising industry is a multibillion-dollar global company that connects producers with customers. According to the research firm eMarkerter, global media advertising spending totaled nearly $629 billion in 2018, with digital advertising accounting for nearly 44% of that amount.For more than a decade, consumers’ perspectives have been shifted in favor of commercials. Advertisements are created based on the preferences of the target audience, and as the population has become more tech-savvy, advertising agencies have shifted their focus from conventional to digital advertising. The internet, as well as the devices, used to access it.Internet advertising has evolved from a risky gamble to the main marketing medium for most businesses. Digital advertising continues to expand by double digits on an annual revenue basis in the United States, with overall spending exceeding $129 billion in 2019.Mobile marketing-Mobile advertising is a form of advertising that uses wireless devices such as smartphones, tablets, and personal digital assistants to view advertisements. In the consumer goods and retail industries, it is extremely necessary.Mobile advertising contents tailored to particular age groups present an opportunity for the mobile advertising industry. The challenges that the mobile advertising industry faces pose a significant risk of new entrants.Content MarketingContent marketing is an old trend that has resurfaced. Many marketers have struggled to determine how powerful banners and display advertising on other people’s content are.Companies are embedding their marketing pitch within the content itself, rather than serving an ad. This can take the form of publisher-tailored content that the advertiser can support or content that the advertiser publishes directly.There are different kinds of businesses and websites that have used content marketing to grow and flourish in the industry. Content marketing is a trend that has contributed a large amount of income to the advertisement industry.To summarise, the advertising industry has evolved through time and will continue to do so as technology advances, allowing advertisers to reach a wider audience and gain a greater understanding of the people to whom they are delivering material.The advertising industry will continue to develop in tandem with innovation. People are also becoming more jaded when it comes to advertisements, pushing businesses to come up with new ways to convey their messages. However, advertisement has a promising future.  '
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[13]:


#18
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='Before we get into the whole discussion, let’s first discuss the basic working of data analysis.Data analysis is defined as a process of cleaning, transforming, and modeling data to discover useful information for business decision-making.Using data analysis, we can extract useful information from the given data and then take corresponding decisions based upon the analyzed data.During the pandemic known as COVID-19, many businesses failed to grow whereas many touched the sky, for example, the transportation of raw materials was drastically low because:The nationwide lockdown was imposed,Low production due to a smaller number of workers,Storage facilities were shut down, and many more for such reasons.On the other hand, new business/startups got a chance to compete in the market by getting early responses from the companies they wanted to tie up with or from the head of the companies that they wanted investments from.The data analysis can help businesses in many ways such as:Using different statistical models that are used in data analysis, the businesses can predict the approximate requirement for the product in the near future and hence they can produce it accordingly.It makes it easier to track the requirement and produce the product accordingly.Because of the lockdown, people had to start working from home, which became a huge advantage for the businesses as they would get quicker responses from their tie-ups.Using data analysis, businesses can create several models and structures to measure the growth of the company and also to make devised plans to increase the revenues and decrease the losses.Data analysis provides different analytical techniques such as:Text AnalysisStatistical AnalysisDiagnostic AnalysisPredictive AnalysisPrescriptive AnalysisUsing these techniques, businesses can analyze everything and predict almost anything.These techniques, tools, and models, can help businesses tackle this horrendous situation that is COVID-19.Blackcoffer Insights 27: SHAILI SHARMA, St. Xaviers College'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[1]:


#19
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text=' Even though COVID-19 has not yet halted and we are facing the nth wave of the coronavirus outbreak across several countries, most notably the US, India, and Brazil. It is a fact that Data Analytics and AI are the big guns of our artillery in this fight against the COVID-19 pandemic. It has helped us in several stages of this outbreak, like the detection of its first outbreak, vaccine development and manufacturing contact tracing, and future hotspot detection. Some of these interesting applications are discussed in this article.A lesser-known fact is that the COVID-19 outbreak was first detected in Toronto, Canada, nearly 7,230 miles away from the first outbreak, nine days before the WHO issued its warning. It was with the help of Big Data Analytics and AI, more specifically Deep Learnings (DL, a subset of Machine Learning) application in Natural Language Processing (NLP) to analyze text inputs that traced the surge of pneumonia cases in the Wuhan province of China. The specialty of DL algorithms is that they mimic the brain cells called neurons and can identify patterns in Big Data. This DL-backed software is used as inputs, reports from public health organizations, global airline ticketing data, etc. These were used to flag unusual surges and potential spreads of infectious diseases.The next application of Big Data Analytics and AI was in the Research and Development of drugs to halt COVID-19. AI was used to analyze the protein structure of the virus, findings that were significant in the progress of vaccine development. In preliminary studies, it was found that it does not mutate as fast as other viruses such as HIV, which means that a prophylactic vaccine is a better way to proceed rather than a therapy. But there is also some evidence supporting the fact that when we find any kind of cure for it, there is a chance of the virus mutating, which is what happened and major mutations have been found in the UK, Brazil, and South Africa. AI also assisted scientists in rapidly shortlisting a set of already available vaccines that could be effective against the coronavirus.Another interesting application of AI can be found in the selection of the right candidates, i.e. most likely to test positive for testing coronavirus in case of insufficient testing resources. This method was first exercised on Greek borders and was called project EVA. Whenever a traveler wanted to come into Greece, he had to fill out a form known as Passenger Locator Form (PLF) at least 24 hours prior to arrival, containing information on their origin country, demographics, point, and date of entry, and the intended destination. EVA then allocated testing resources according to the size of the set of passengers to be tested. After the test results, if found positive, they are put in quarantine. The results were sent back to the program for real-time learning.The question remains how EVA made allocations, It was found that, statistically, only the origin country and the city were significant factors for screening. Ultimately, from a variety of countries and city pairs, EVA had to predict how many testing resources were to be allocated at each entry point and to particular passengers from a location is technically called the Multi-Armed Bandit (MAB) problem, and the chosen method to solve this problem was an AI algorithm called optimistic Gittins index. This algorithm identified on average 1.85x as many asymptomatic, infected travelers as random surveillance testing, and up to 2-4x as many during peak travel. After the test results, if found positive, they are put in quarantine. Following the collection of significant data through the aforementioned process, after a certain period, policies were made categorizing them separately and imposing restrictions on travelers from the specific location. This EVA as presented above was in operation from August 6th to November 1st processing around 38,500 PLFs each day and testing on an average 18.5% of households entering the country every day.Above mentioned applications just show the tip of the iceberg and there is more to get into some of the other developments to watch for include the use of Image Recognition to identify covid based on x-ray images, the use of Deep Learning to predict the 3-D protein structure associated with COVID-19 and so on.Blackcoffer Insights 27: Aniruddha Surse, NIT Nagpur'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[2]:


#20
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='“Data is the new oil” has become the most important trendline of the 21st century. The reason for this is the advancements in the fields related to data analysis. The field of AI Machine Learning Statistics and Data Mining all deal with data and are developing at such a staggering pace, that these fields have become the most popular buzzwords these days.Buzzwords are originated through technical terms but often the underlying essence is ignored through fashionable use and mainly used to impress. This is the main reason for the misconception amongst people. AI, ML, Stats, Data Mining, and many other fields related to the analysis of data are most often mistook for one and the same thing, thus all these words are often used interchangeably to convey one and the same thing. But this is not true at all!The only similarity between these disciplines is that all of these disciplines are related to the analysis of the data and converting this data into “information”.In academics, while learning AI Machine Learning Statistics and Data Mining, the academic approach only wander in the technical definitions and concepts but the underlying essence and the aim of the discipline remain unexplored, same is the case with most of the articles out there which try to explain the difference. Thus, becoming the primary cause of confusion between learners. Hence, this article explains the difference by explaining the philosophy and the aims of each of the disciplines rather than wandering in the technical definitions.Then what is the difference between AI Machine Learning Statistics and Data Mining?The essential difference between these disciplines lies in their “aims” and the approach taken to achieve those aims.The aims of each of these fields are explained in detail in the following sections:Artificial Intelligence:The aim of Artificial intelligence is to understand intelligent entities. Then to satisfy this aim, we must first understand what is intelligence and what makes an agent intelligent agent? The answer to all these questions can be found by studying intelligent agents and the best example of an intelligent agent can be found by just standing in front of a mirror! Yes! Humans are the best examples of intelligent entities. Thus, in the 1900s, researchers began exploring the thought process, the reasoning process of humans as human beings were considered as an ideal intelligent agent. Mimicking human behavior became the aim of AI in the initial years of the research. After setting this goal, studies and experiments began and the most famous experiment conducted in the initial years to achieve this aim was the Turing Test! Turing defined intelligent behavior as the ability to achieve human-level performance in all cognitive tasks, sufficient to fool an interrogator. But this test received criticism as only mimicking human behavior is not exactly intelligence. Because intelligence should be related to the working of the human brain as without the human brain, intelligence has no meaning!            Thus, mimicking human thought processes and reasoning became the transformed aim. The field of Psychology and philosophy also resonate with this aim that is to understand the human thought process. The difference is that AI not only tries to understand the thought process but to mimic it, build it. The collaboration of these fields resulted in models such as Neural Networks which try to mimic the function of neurons present in the human brain. So, basically, this initial aim was human-centered and humans were considered as the ideal intelligent agent.Concurrently, the field of computer science was developing at a greater pace. With the advances in computer science, the experiments and theories could be easily tested and validated. As programs were being applied to solve real-life problems, it was found that computers performed better than humans at some tasks that are really complex for humans. One of the best examples of this could be the chess-playing program. An AI program defeated the world’s best chess player Garry Kasparov. This incident indicated that human intelligence is not the ultimate intelligence or else a human would have been able to defeat the AI program. This leads to a question, is human intelligence the ideal intelligence?As computers became more advanced, they proved to be better than humans at certain complex tasks. That is why the new definition of intelligence was being related to the ability to solve cognitive tasks or problems. So, rather than considering the nature of agents, researchers began to study the nature of intelligence itself. Then the question comes how to test or validate intelligence? The best way to test intelligence is to solve cognitive problems. An agent can be said intelligent only if it can solve a complex problem. The problem-solving approach can be easily tested and validated on computers. Thus, some researchers began studying the ideal intelligence, and the selected agent to validate the experiments was the computer. So, a computer and problem-solving approach were adopted. So, the human-centered approach and computer, problem-solving approach are the two main aims of AI. Both of these fields have contributed to the field by giving valuable insights.Both the aims are important and both of these collaboratively form the main aim of AI!Machine Learning:In the problem-solving approach, there is a big challenge that AI has to overcome in order to achieve its aim. Consider the example of solving a math problem. There are two cases by which intelligence can be tested in this problem-solving approach. Let us say two math problems are given for you to solve. The first problem is familiar to you and the second problem is not.Consider the first problem. The first problem is familiar to you, that means you know how to solve such kind of problems as you have already solved some similar problems in the past. So, there comes a question, how our mind is able to solve that problem? The answer is, that you have solved similar problems in the past, thus you have learned from the past data, how to solve such problems, thus even if you haven’t seen that problem in the past, you will still be able to solve similar problems. This is one form of intelligence.Consider another case where you are given a second problem where you have not solved such kind of problems in the past. Then to solve this problem, you will try to consciously gather and manipulate the given information so that you reach a certain conclusion. This kind of approach does not necessarily rely on the past data but completely on the reasoning process. This is the second kind of intelligence.For AI to build intelligent agents, both of these kinds of intelligence must be developed in the agent. But, the reality is that AI has reached the point where it is able to build agents which can only learn from past data and find some useful information. AI today has not reached a point where it can build agents who can think on their own. That is the second type of intelligence.So, the way AI is able to implement the first type of intelligence is through Machine Learning! So, the domain of AI which focuses solely on implementing the first kind of intelligence is in fact Machine Learning. That is the reason why ML is called the subset of AI! So, this is the main difference between AI and ML.Technically speaking, “It is the field of study that gives computers the ability to learn from past data and find some meaningful conclusions, patterns without being explicitly programmed”. This statement needs some elaboration. The essence of ML is related to the process of “Generalization” and learning from past data. Generalization is an abstraction by which common properties of specific instances are formulated as general concepts or claims. Consider how we humans recognize daily life objects. If we see an animal, then we can easily recognize if it is a “dog” or a “cat”. It is a very trivial task for us but have you ever wondered how our mind is able to do it? The answer is Generalization!If you were given a picture of a dog, you can easily recognize that it is a picture of a dog, because, our minds have abstracted the description of a dog and formulated it into a “concept” of what a dog is and these concepts became better and better as we learned from the past experiences of a dog. So, the way we think is dependent on the fact that things are represented as generalized concepts in our minds.With generalization only, can come real “information”. So, we try to give computers the ability to generalize the “raw data” and convert it into “information” which can be patterns and trends in the data on their own.Statistics:This is the discipline that concerns the collection, organization, analysis, interpretation, and presentation of data. Statistics tries to deal with data with the only aim that is to explain it. So, it is the study of explaining the data itself! Statistics has two main domains which are Descriptive statistics and Inferential statistics.Descriptive statistics deals with the explanation or description of data thus the name “Descriptive” statistics. It tries to explain as much information as possible, easily about the whole large data which would be a very complex task otherwise. Inferential statistics try to make accurate inferences from the available small data. We use inference in many tasks in our daily lives. Consider a simple case of cooking a soup. After completing the recipe, you will taste a small sample, that is, a spoonful of the soup to check if the soup tastes good or bad. Depending upon the result of the sample, you make an inference about the whole soup that if the soup as a whole, good or bad. Similarly, in statistics, there are cases where you have to apply inference to have meaningful information. For example, consider a case where you cannot gather the whole data because it is very time-consuming and costly. In these cases, applying inference based on the available sample introduces uncertainty. That is where inferential statistics come for help.So, the use of data in the context of uncertainty and decision-making in the face of uncertainty is what statistics deals with. So, however, and whatever the data, statistics tries to explain that data. This aim does not resonate with that of AI and ML, but statistics help these fields to correctly interpret the data!Data Mining:  “Data”, is not useful at all in its raw form. Consider examples of sensors used in industrial applications. These sensors might be used in a manufacturing plant to sense different properties like temperature, pressure, etc. The raw data generated by these sensors are not useful until and unless it is converted to a suitable form, then processed, analyzed to gather valuable insights, which can be used to solve a problem!Due to its unique aim of capturing the essence of very large datasets, to gather insights, Data Mining is also referred to as “Knowledge Discovery”. That is why, Carly Fiorina, former CEO of Hewlett-Packard once said, “The goal is to turn data into information and information into Insight”. This statement completely explains the aim of Data Mining!            So, the difference between AI Machine Learning Statistics and Data Mining lies in their aims. But the approaches taken in all of these fields, help in one way or the other in fulfilling the aims of the other fields. This is the beauty of these fields!Blackcoffer Insights 25: Abhishek Govekar, Vishwakarma Institute of Technology, Pune, India.'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[3]:


#21
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='Data Science is gaining popularity exponentially over the past decade, and thanks to that we are now enjoying better products, recommendations, and smoother life. Data science is an interdisciplinary subject that includes statistics, math, IT, etc.Now there is so much to do in Data Science,  so we need an arrangement where all this can be accessible in one place. It will be very hectic to go to hundreds of different resources while doing analysis or building models. But don’t worry, PYTHON is there for you.Yes, you read it right, Python is a general programming language that can provide everything you need for Data Science. Several features that have made Python become the choice of data science in past times are:1. Python is a progressively typed language, so the variables are defined automatically.2. Python is more readable and uses lesser code to play out a similar task when contrasted with other programming languages.3. Python is specifically typed. In this way, developers need to cast types manually.4. Python is an interpreted language. This implies the program need not have complied.5. Python is flexible, convenient, and can run on any platform effectively. It is adaptable and can be integrated with other third-party software effectively.Now let’s see why Python become the choice of data science:PANDASThis library available in python makes it very easier to analyze the data, you can read a variety of data sets like CSV, XML, XLSX, JSON, etc. You can perform several operations like Groupby, sorting with the help of easily accessible objects from pandas.NUMPYThis package helps you with any numerical operation that is needed to be performed in Data science, for example calculating Euclidean distance, finding ranks of the matrix, etc.matplotlib AND SEABORNThese are excellent data visualization libraries available in python that produce some excellent visualization like shown here,sklearnIt provides you state of the art machine learning algorithms for your accurate predictive analysis. Scikit–Learn is characterized by a clean, uniform, and streamlined API, as well as by very useful and complete online documentation. It provides a selection of efficient tools for machine learning and statistical modeling including classification, regression, clustering, and dimensionality reduction via a consistent interface in Python.KERASKeras is an open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library. Keras is an industry-strength framework that can scale to large clusters of GPUs or an entire TPU pod.See, Now that you have a variety of resources available here in python, then why go anywhere else. Python I also becoming the world’s most loved and most wanted programming language and it will surely help to get you the job. Data science consulting organizations are empowering their group of developers and data scientists to utilize Python as a programming language. Python has gotten well-known and the most significant programming language in an extremely brief timeframe. Data Scientists need to manage a large amount of data known as big data. With simple utilization and a huge arrangement of python libraries, Python has become a popular choice to deal with big data.Seeing the stats we can clearly see that Python has taken over other Languages needed for data science. It has also surpassed R which is exclusively built for Data science. Isn’t this exciting.Python in Data science has empowered data scientists to accomplish more in less time. Python is an adaptable programming language that can be effectively understood and is exceptionally amazing as well.Python is highly adaptable and can work in any environment effectively. Additionally, with negligible changes, it can run on any operating system and can be integrated with other programming languages. These qualities have settled on Python as the top choice for developers & data scientists.So next time you do analysis or work on any Data Science project, feel proud cause you are working with the most loved language of the world.Blackcoffer Insights 25: Divyansh Bobade, LNCT (Bhopal)'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[7]:


#22
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='It was in March that Tech giant Google came up with a ground-breaking announcement of Google Fit able to measure one’s heart and respiratory rates using their smartphones. This news spread like wildfire. Instantly became the talk of the town. This feature was said to be available to the Google Fit app exclusively to its Pixel phone users. Google also plans to expand to its other Android devices in the future.This was after Google’s newest endeavor of acquiring the Fitbit for a whopping $2.1 billion. This acquisition not only does steps up a stage for a potential Google Smartwatch but also gives Google the ownerships to Fitbit’s health business and wealth of data assets. Users just need to place their head and upper torso in view of the front-facing phone camera for those who wish to measure their respiratory rate. For measuring Heart rate the user just has to place their finger on the rear-facing camera lens. Mind-blowing right!Once the measurements have been taken the users simply have to store and save them in the Google Fit app to monitor and track their day-to-day wellness. On asked how it’s measuring these heart rate and respiratory rate, Google Health director of health technologies Shwethak Patel explained that these features rely on the sensors that have been built into the smartphone, such as its camera, microphone, and accelerometer. Thanks to increasingly power sensors even in affordable smartphones and advancements in computer vision, these features let use one smartphone’s camera to track even tiny physical signals like your chest movement to measure your respiratory rate and subtle changes in the color of your finger for your heart rate.Pixel underwent and completed initial clinical trials to validate the algorithm cloud work in a variety of different world conditions and that too with many people while developing the features. Since our heart rate relies on approximating blood flow from color changes in someone’s fingertip, it has to account for factors such as lighting, skin tone, age. Adding to be able to measure heart and respiratory rate soon Google Fit also displays user daily stats such as daily goals, weekly goals, heart points, workout, and also sleep monitor.Blackcoffer Insights 25: Sri Vishnu S, Kristu Jayanti College (Bangalore)'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[6]:


#23
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='Tech Giants such as Apple and Google have launched their own augmented reality kits — AR Kit and AR Core resp to help app developers to create high-quality mobile apps. As for VR, the latest buzzword associated with it is v-commerce. One of the examples is Alibaba, which in November of 2016 introduced VR shopping to its customers across China.Blackcoffer Insights 24: Nikhil Singh, St.peters college agrae and my two cents on what mobile engineers could do to prepare themself for the upcoming challenge! and the future of mobile apps.Coding Bootcamps Increase the Supply Side for mobile apps  DevelopersAlthough the demand for mobile developers has not fallen sharply compared with previous years, the growth rate has slowed. On the other hand, owing to a great number of coding boot camps that target to bring up mobile and frontend engineers, a great number of developers are flooding into the developer job market.The supply side for mobile engineers has increased rapidly in recent years, raising the standards for qualified mobile engineers. The days of a mobile engineer easily getting over 10 job offers are over.Recode ran an article in mid-2016, that begins: “The mobile app boom kicked off in July 2008, when Apple introduced the App Store. Now it is over.”According to data gathered and analyzed by the CakeResume team, JavaScript and Python are now the most in-demand programming languages for companies who post product development jobs on CakeResume, both together taking up almost half of the job opportunities, while mobile engineers taking up merely 10% of the job opportunities.learn more about tech salaries in TaiwanTech Salaries in Taiwan 2019OverviewPeople Stop Downloading mobile appsWhile App usage continues to grow and revenue also ascends, the majority of consumers actually download zero apps per monthlet’s face the truth, born and raised up in the era of the information explosion, individuals don’t have the leisure to check on what’s new on the app store every day. When it comes to choosing an App to download, many people feel overwhelmed by the sheer number of options available.Take the productivity category of the App Store as an example, there are over thousands of Apps in this category, and new Apps popping up almost every week. It is almost impossible for users to dig through and learn all the apps. They just choose what is on top then click Install.And not to mention that if you could build an awesome productivity app, Google and Apple could do it 10 times better and faster than you. In the end, there’s not as much need for individual apps to accomplish similar convenience factors.What You Could Do to Prepare for The future Changes for mobile appsAlthough the bull run for mobile development has ended, there are still over 2 billion smartphone users, 27.5 billion mobile apps downloaded, and time spent per day on mobile devices has increased rapidly in recent years. Moreover, apps offer a user experience that even ‘Responsive Websites’ are unable to provide.In order to stay on top of the game, you will have to differentiate yourself from the pack — other mobile engineers. If you are a mobile engineer who only knows how to code and tweak your App’s UI, you aren’t that different from others after all. Below I listed a few ways that you could implement so that you could stay competitive in the job market.1. Choose Newer Technologies When You Have the ChanceIf you are an Android developer, you could start choosing a job involving Kotlin or Flutter, which are backed by Google. If you are an iOS developer, try to look for a job to get involved with using Swift.Although older mobile coding languages, such as Java and Objective-C, still have their upsides, the main reason why Kotlin and Swift are created at first is to address the older languages issue. It means that Kotlin and swift provide many safety mechanisms available out-of-the-box while being more concise and expressive than Java and Objective-C at the same time.Kotlin vs. Java: Which One You Should Choose for Your Next Android AppA detailed comparison of Java and Kotlin to help you decide which language will work best for your next mobile… Objective-C vs Swift in 2019The article covers a brief comparison between Objective-C and Swift in 2019, with ABI Stability.2. Build Up Your Domain Knowledge and Industry SkillsThis point is especially crucial. Each industry has its specific skills to learn and refine. For example, if you are building a stock, foreign exchange, future, or options market app, you will have to understand the WebSocket protocol, which lets you transfer as much data as you like without incurring the overhead associated with traditional HTTP requests.For the streaming media industry, possessing experience with live streaming protocols such as HLS, RTMP, WebRTC will be a must to deal with streaming-related apps.These industry skills can really make you stand out, and adds another layer of expertise to your already pretty impressive mobile engineer title, making you more valuable in the job market.3. Follow the Irresistible Tech TrendThe future of mobile app development will be shaped by how businesses harness mobile technology to solve people’s everyday problems.Don’t try to fight the irresistible trend, see how you can surf the trend! Exposing yourself or trying to get a job in the field mentioned below will absolutely give you more opportunities as a mobile developer.So what are the trends for mobile apps development?IoT (Internet of Things)“The driving force behind the growing mobility market is the impact of the IoT and its broad reach,” writes the experts at Maryville University online. In the future, apps will need to speak to each other, in the same way, that devices in the IoT communicate. The market needs companies that can develop custom IoT applications — sensors and devices, web apps, and both B2B and B2C mobile end-user apps.AI (Artificial Intelligence)According to experts at Maryville University, AI is everywhere, from predictive analytics algorithms used by retailers like Target and Amazon to anticipate shopping needs to fraud detection monitors from banks and credit cards. Some more examples: Tinder uses machine learning to increase a user’s chances to find a match. Google Maps makes the process of choosing a parking spot easier.AR (Augmented Reality) and VR (Virtual Reality)'
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[1]:


#24
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='AI allows those in training to go through naturalistic simulations in a way that simple computer-driven algorithms cannot. The advent of natural speech and the ability of an AI  computer to draw instantly on a large database of scenarios means the response to questions, decisions, or advice from a trainee can challenge in a way that humans cannot in health and medicineHealth monitoring:Wearable health trackers-like those from FitBit, Apple, Garmin, and others- monitor health rate and activity levels. They can send alerts to the user to get more exercise and can share this information with doctors.Technology applications and apps encourage healthier behavior in individuals and help with the proactive management of a healthy lifestyle.AI increases the ability for healthcare professionals to better understand the day-to-day patterns and needs of the people they care, for better feedback, guidance, and support.Medical imaging: Machine learning algorithms can process unimaginable amounts of information in the blink of an eye and provide more precision than humans in spotting even the smallest detail in medical imaging.  A few of them are Blackford, Zebra, Enlitic, Lunit.The company “Zebra Medical Vision” developed a new platform called profound, which analyzes all types of medical imaging reports that are able to find every sign of potential conditions such as osteoporosis, aortic aneurysms, and many more with a 90% accuracy rate.Digital consultation: For eg, the digital health firm #HealthTap developed “Dr. AI”, and apps like Babylon in the UK use AI to give medical consultations based on personal medical history and common medical knowledge. Users report their symptoms into the app, which uses speech recognition to compare against a database of illness and asks patients to specify symptoms to triage whether they should go to the ED, urgent care, or a primary care doctor.AI robot-assisted surgery: Robots have been used in medicine for more than 30 years. Surgical robots can either aid a human surgeon or execute operations by themselves. They’re also used in hospitals and labs for repetitive tasks, in rehabilitation, physical therapy, and in support of those with long-term conditions.      Health chatbots such as Babylon, Ada, and mostly close-ended communications.Machine learning is a type of AI that allows computers to make predictions without being explicitlyProvides formal conceptual Framework for input processing and decision making in diagnosis and managementObjective decision making with less varianceHigh speed and efficiencyUnlock the power of big data and gain insight into patients.Support evidence-based decision-making, improving quality, safety, and efficiency.Coordination care and faster communicationImprove patient experiences and outcomesDeliver value and reduce costsImprove health system performance and optimizationExamples of AI in healthcare and medicine#MICROSOFT: Predictive analysis in vision care#GOOGLE: clinical decision support in breast cancer diagnosis#IBM WATSON: precision medicine in population health managementActionable medical insights:  An ever-increasing amount of medical data are being digitized at all public and private healthcare institutions. However, by its very nature, this kind of data is messy and unstructured. Unlike other types of business data, where traditional statistical methods can be used for quick insights, patient data is not particularly amenable to simple modeling and analytics tools. For eg.- Enlitic, a San Francisco-based start-up, has a mission of mixing intelligence with empathy and leverage the power of AI in health and medicine for precisely generating.Therefore, a massive parallel effort to rationalize the legal and policy-making is needed to bring the full benefit of advancement in AI technologies into the healthcare space. As technologies and AI/ML enthusiasts, we can only hope for such a bright future where the power of this intelligence.Blackcoffer Insights 24: Shaffy Garg, Baba Farid college of management and technology'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[2]:


#25
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text=' In today’s world, telemedicine technology is one of those technologies which has brought about a change. Compared to the early days there have been remarkable differences in the methods of consultation with a doctor. In the years that have passed by, consultation for a disease with a doctor was quite hectic. It involved waiting, traveling, etc. But with the advent of telemedicine opportunities, this has completely changed.It is a rural area that has been completely blessed with the invention of telemedicine. Today a considerable amount of people are able to consult doctors remotely. Not just doctors, but specialists in various fields of medicine. This has been of great importance as far as rural people are concerned. There are a lot of telemedicine tools that have been found. There are a lot of areas like ophthalmology, oncology, dermatology, etc where the facility of telemedicine has been practiced.Most of the patients are truly benefitting from telemedicine. Patients are pretty satisfied with the consultation they are getting. They don’t have to travel now. They can consult doctors and other specialists from remote areas. The cost of consultation has become pretty much affordable. Moreover, they get exposed to highly efficient and qualified experts in the field of medicine.On the other hand, there can be patients who do not fully get satisfied through a virtual consultation. Rather they might feel that they need to have a direct talk with the doctor, which can boost up their confidence and also it helps to maintain a better relationship with the doctor and patient. The patients feel trust when they talk to doctors face to face. Moreover, doctors also might be able to console their patients when they have direct interaction with their patients. There could also be patients who doubt if these virtual methodologies are really trusted worthy or not. It is so because, while the patients have direct contact with their patients, the qualifications are visible to the patients. So there shall be no question of distrust.There are major challenges like better connectivity of the internet, without which the patients will not be able to have continuous interaction with the doctors. Technical glitches may hinder the consultation too. Moreover, emergency situations cannot be addressed beyond a limit through telemedicine.Blackcoffer Insights 23: Miriam sam, Scms school of technology and management.'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[3]:


#26
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='TECHNOLOGY is playing a dominant role in human life. And as in our daily lifestyle technology is used each and every place where human beings are present. so How we forecast future technologies? So, the fact is that technology is not a single immutable piece of hardware orbit chemistry. It is simply knowledge of physical relationships and it systematically applied to the useful arts. This knowledge can vary continuously over time. It can range basic phenomenon can be applied to an end product, device, or production machine in a mature operating system. Even as time passes, the performance characteristics of any machine, product, or operating system are normally improved in small continuous increments over time.Advance in technology is usually nothing more than an accumulation of small advances not worth introducing individually to make a significant change in total technology. Forecast future technologies no matter how accurate – unless they eventually influence action.Blackcoffer Insights 23: Arvind Pal, PACIFIC SCHOOL OF ENGINEERING, SURAT'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[4]:


#27
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='In this era, everyone is busy in his life. No one spares time for someone. so robots tackle life loneliness? We have no time to stand and stare. This causes loneliness. Loneliness is due to fast-moving life and hurry and worry. In the 21st century, everyone is spending most of their time earning money. They are day-by-day transforming into money-earning machines. Due to his perception, we are losing our relations as we can’t give enough time to our families. They are busy earning their livelihood. They never get satisfies and always want more. Everyone is in a race of earning and then sends his children far away from himself to make them well settled in order to maintain their status. But one faces the consequences of this kind of lifestyle in his old age. People usually don’t spend time with their children when their children are in their childhood but they yearn for the company of their children in their old age. They feel lonely but have none to talk with. They are still attached to people, places, belongings, and memorable events from the past, although they understand that life cannot continue the same way as earlier, and this may easily result in feelings of loneliness and social isolation. As they become older, they need to be attended to in order to meet their daily needs. Robots can indeed serve the purpose but they can aid only mechanical care. They are just emotionless, feeling fewer entities. Robots aren’t creative. They can work only according to a pre-programmed system and till now no such software has been created that can transfer human feelings into robots. This is the major loophole of using robots to tackle late-life loneliness. They can undoubtedly serve their masters but their masters can’t share their feelings with them as they are incapable to understand or react to them. They can’t change their activity according to the occasion. They can provide anything except emotional satisfaction and without emotional gratification, loneliness can’t be tackled. A master can be attached to his robot but that attachment is only due to their service. They can’t share their thoughts with them. In this era, we are rendered lonely and in late- life feel the need for a companion with whom we can talk. We feel a need for someone who can listen to us, react, and advise us to tackle our day-to-day life problems. But we can’t find one. We have, undoubtedly, achieved great progress in terms of technology but we can never fill the void in the heart of a person struggling with his late-life loneliness with our advancements. We are in a kind of mental trance in which we experience fame, progress, wealth, etc. but when we gain our consciousness, we find ourselves lonely in this world. Relations in life are actual wealth in a person’s life. We can never enjoy such a bond with a robot. It will follow our commands, take our appropriate care but can’t react to our emotions. It can’t console us. When a man lives in loneliness for a long time, it eats up his conscience and transforms him into a machine. Robots can never become our friends, crack jokes, weep our tears, or establish an emotional connection with us. They can’t understand our feelings. Many people go into depression. Depression or the occurrence of depressive symptoms is a prominent condition amongst older people, with a significant impact on their well-being and quality of life. They remain sick and loneliness directly impacts longevity. Lonely people often think that they are no longer needed in this world and thus they want to die. It impacts their mental, psychological, social, and physical health. Robots prove to be useless in these matters. They can provide motivation, only if they have the software to do so but can’t, themselves, react to such a situation. They don’t know about anger, happiness, or sadness. They don’t themselves bring food when their master is hungry until commanded to do so. They can’t even offer a glass of water themselves. A man living without relations and without fellow feeling no longer remains a human being. He becomes none less than a machine as, due to his loneliness, he becomes mentally ill and goes into a condition like trauma where he no longer enjoys nature’s blessings, becomes happy or sad as he loses the ability to react and hence can’t act. It’s a dangerous situation. We face a plethora of challenges in assisted living facilities such as not being addressed to emotional needs, being neglected, and forcing a withdrawal from social activities. Relations in life, interaction with fellow beings, sharing of joy and happiness, and fellow feeling make a man from mere a ‘being’ to a ‘social being’ and finally a human being.Blackcoffer Insights 22: Ishatpreet Singh, IISER (Mohali)'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[5]:


#28
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='Care Robots, as the name suggests, are robots that are used for hospitality purposes like fetching water, cracking jokes and keeping a patient in good harmony, etc.The senior care industry has been at the forefront for quite a period. The reason being, Nuclear families becoming very busy in their schedule that one day when the parents of this nuclear family are old, children are nowhere to take care of. Instead, nursing homes have become a trend.  Parents with mental illness are being put in rehab.As observed for a few decades robots are taking care of many activities and they are even dominating a few fields like the automobile industry. So, Machines can indeed help humankind in achieving remarkable success in many fields.Do senior citizens love machines ?, You should look at them when they watch TV when they love to play video games on a smartphone. So, If we give them a human-like structure with the ability of a smartphone to make these people happy then it would be quite a monstrous feat.Blackcoffer Insights 22:Sadhana Kanaparthi,  KIAMS'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[6]:


#29
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='Management acts as a guide to a group of people working in the organization and coordinating their efforts, towards the attainment of the common objective.Management challenges for future digitalization of healthcare servicesCHARACTERISTICS for future digitalization of healthcareUniversal: All organizations, whether it is profit-making or not, require management, for managing their activities. Hence it is universal in nature.Goal-Oriented: Every organization is set up with a predetermined objective and management helps in reaching those goals timely, and smoothly.Continuous Process: It is an ongoing process that tends to persist as long as the organization exists. It is required in every sphere of the organization whether it is production, human resource, finance, or marketing.Multi-dimensional: Management is not confined to the administration of people only, but it also manages work, processes, and operations, which makes it a multi-disciplinary activity.Group activity: An organization consists of various members who have different needs, expectations, and beliefs. Every person joins the organization with a different motive, but after becoming a part of the organization they work for achieving the same goal. It requires supervision, teamwork, and coordination, and in this way, management comes into the picture.LEVELS OF MANAGEMENTTop-Level Management: This is the highest level in the organizational hierarchy, which includes the Board of Directors and Chief Executives. They are responsible for defining the objectives, formulating plans, strategies, and policies.Middle-Level Management: It is the second and most important level in the corporate ladder, as it creates a link between the top and lower-level management. It includes departmental and division heads and managers who are responsible for implementing and controlling plans and strategies which are formulated by the top executives.Lower Level Management: Otherwise called functional or operational level management. It includes first-line managers, foremen, supervisors. As lower-level management directly interacts with the workers, it plays a crucial role in the organization because it helps in reducing wastage and idle time of the workers, improving the quality and quantity of output.future digitalization of healthcare    In the current day and age, a wave of digitization has taken over the world. All the emerging technologies like Artificial Intelligence and others are helping people live better and easier life. The service sector has also benefited a lot from digitization. It got further boost when Prime Minister Narendra Modi, in the year 2015, launched a campaign known as ‘Digital India.’ Among the various industries in the service sector, digitization has had a massive impact on the operation of the healthcare and diagnostic industry. It has helped in the development of this industry and enhances life for millions of people. The condition of India in terms of healthcare has been quite grim. Patients who need attention are neglected and are unable to avail proper treatment or diagnosis. However, with the coming up of digitization, there has been hoped for this industry also.According to Dr. Keshab Panda, CEO & MD, L&T Technology Services, there are three trends emerging in the healthcare and diagnostics ecosystem as a result of digitization.1. Value-based healthcareThis is a model of healthcare where doctors and hospitals are paid based on patient health outcomes. The coming up of digital tools in this segment of healthcare can be considered as the starting tool. Dr. Panda further says that the implementation of advanced digital technologies in patient-care domains like- mobile health apps, telehealth, wearables, and remote monitoring can help in enhancing accessibility, boost efficiency and augment the effectiveness of treatment and preventive care.2. New product developmentThe healthcare industry over the past few decades has changed drastically. With the coming of emerging technologies, new surgical procedures and medical devices have been made available.3. ConnectivityWhile talking about connectivity in healthcare, Dr. Panda said that with digitization taking over the industry, it is only a matter of time when with the help of the internet and smartphones, digital healthcare will be everywhere. The internet for once has helped doctors connect to their patients and also with one another. This has helped in enhancing the doctor-patient engagement by increasing their interaction time.Big data aggregates information about a business through formats such as social media, eCommerce, online transactions, and financial transactions, and identifies patterns and trends for future use.For the healthcare industry, big data can provide several important benefits, including:Lower rate of medication errors – through patient record analysis, the software can flag any inconsistencies between a patient’s health and drug prescriptions, alerting health professionals and patients when there is a potential risk of a medication error.Facilitating Preventive Care – a high volume of people stepping into emergency rooms are recurring patients also called “frequent flyers.” They can account for up to 28% of visits. Big data analysis could identify these people and create preventive plans to keep them from returning.More Accurate Staffing – big data’s predictive analysis could help hospitals and clinics estimate future admission rates, which helps these facilities allocate the proper staff to deal with patients. This saves money and reduces emergency room wait times when a facility is understaffed.Challenges  for future digitalization of healthcare 1. CybersecurityAlthough ransomware, data breaches, and other cybersecurity concerns are nothing new to the healthcare industry, the 2020 Covid-19 pandemic revealed just how vulnerable sensitive patient health information really is.The recent growth of digital health initiatives- like telehealth doctor visits — is a major contributor to the severe increase in breached patient records. As more healthcare functions continue to move online over the next year, it’s extremely important to ensure these processes are protected from outside threats.2. Invoicing and Payment Processing:Medical practices are citing patient collections as their top revenue cycle management struggle as patients are becoming responsible for a larger portion of their medical bills. In order to help encourage patients to submit payments in a timely manner, providers must adhere to patient payment preferences.To meet patient expectations and improve the user experience, ensure billing statements are patient-friendly. You should offer paperless statements and a variety of payment options (e.g. credit card, etc.) via an online patient portal and utilize the latest payment technologies, such as mobile and text-to-pay. New features like text or email reminders help effectively communicate with patients and encourage them to pay their financial obligations.3. Patient ExperienceThe medical insurance landscape has experienced some significant changes in recent years. As more patients are responsible for a larger portion of their healthcare bill, they naturally demand better services from their providers.Healthcare organizations will face tougher competition in attracting and retaining patients who demand an experience that matches the level of customer service they expect from other consumer brands.They demand a streamlined patient experience so they can “self-service” to resolve most questions, issues, or concerns (e.g., downloading an immunization record, booking an appointment, paying their bills, or checking their account/insurance status) whenever, wherever, and however is most convenient for them.Blackcoffer Insights 22: PRIYAM VERMA AND RIYA MALIK'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[7]:


#30
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
text='One thought always comes to my mind…What if we lived without that dander hanging over our heads?And you all know what is that danger is. For more than 70 years the world has faced the very real threat of nuclear war.Nuclear war is a real and growing threat. The United States and Russia have left critical agreements and treaties, while actively planning to add new types of weapons to their arsenals.Meanwhile, US nuclear policy remains rooted in the Cold War, increasing the risk that nuclear weapons could be used again.It doesn’t have to be this way. With the right policy changes and a commitment to diplomacy, the United States can be a leader in reducing the nuclear threat and we can also help them. But the question arises is what are the activities are being done for the prevention of nuclear holocaust?And they are :Pressuring Congress to support change,Holding the White House accountable through independent research and analysis,Increasing public demand for changing nuclear weapon policies.Now the question is what can we do to prevent a nuclear holocaust?Tell government: Don’t fund a nuclear arms race.Call for investment in public health and public security, not nuclear weapons.Tell government: The United States should never start a nuclear war.Urge presidential candidates to make preventing nuclear war a priority. ACTIVIST RESOURCESnuclear weapons are extremely strong to show the power of a nation. But no one knows the drawback of this attack between the two countries. For example Hiroshima and Nagasaki, the realistic bomb blast explained the blood of each people. The crux of my presentation is that the circumstances surrounding nuclear war call for a new level of commitment by the scientific community to reduce the risk of nuclear war. To generate new options for decreasing the risk, we need analytical work by people who know the weaponry and its military uses. But that is far from enough to do the job. We also need scholars who know the superpowers in-depth; people who know other nuclear powers; people who know third-world flashpoints; people who know international relations very broadly; people who know a lot about policy formation and implementation, especially in the superpowers; people who understand human behavior under stress, especially leadership under stress; people who understand negotiation and conflict resolution people and much more. In other words, the relevant knowledge and skills cut right across all the physical, biological, behavioral, social. International Physicians for the Prevention of Nuclear War, World Health, British Medical Association, and the American Medical Association; others have contributed as well. The central point is that a kind of awakening has occurred in the medical community to the responsibility of addressing the immense nature of the threat to public health. At last, I want to conclude that: “ In a nuclear war, except evil force, no one is the winner. Science and humanity become the villain. Everyone knows that but the gamblers want to play their cards. Be aware of the nuclear gamblers “.I hope that soon the world will be full of love and affection and there will be no place for hate and the wars will be held.Blackcoffer Insights 21: Rutika Pagar, C.M.C.S College Nashik'
sia=SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
positive_score = sentiment_scores['pos']
negative_score = sentiment_scores['neg']
# Polarity and subjectivity scores
polarity_score = sentiment_scores['compound']
subjectivity_score = sia.polarity_scores(text)['neu']

# Text statistics
word_count = lexicon_count(text)
avg_sentence_length = avg_sentence_length(text)
complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
percentage_complex_words = (complex_word_count / word_count) * 100
fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

# Additional metrics
avg_words_per_sentence = word_count / sentence_count(text)
personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count

# Print the results
print("Positive Score:", positive_score)
print("Negative Score:", negative_score)
print("Polarity Score:", polarity_score)
print("Subjectivity Score:", subjectivity_score)
print("Avg Sentence Length:", avg_sentence_length)
print("Percentage of Complex Words:", percentage_complex_words)
print("FOG Index:", fog_index)
print("Avg Number of Words per Sentence:", avg_words_per_sentence)
print("Complex Word Count:", complex_word_count)
print("Word Count:", word_count)
print("Syllables per Word:", avg_syllables_per_word(text))
print("Personal Pronouns:", personal_pronouns)
print("Avg Word Length:", avg_word_length)


# In[9]:


import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
def analysis(text):
    sia=SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    positive_score = sentiment_scores['pos']
    negative_score = sentiment_scores['neg']
    # Polarity and subjectivity scores
    polarity_score = sentiment_scores['compound']
    subjectivity_score = sia.polarity_scores(text)['neu']

    # Text statistics
    word_count = lexicon_count(text)
    avg_sentence_length = avg_sentence_length(text)
    complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
    percentage_complex_words = (complex_word_count / word_count) * 100
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

    # Additional metrics
    avg_words_per_sentence = word_count / sentence_count(text)
    personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
    avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count
    return sentiment_scores, positive_score, negative_score, polarity_score, subjectivity_score, word_count, avg_sentence_length, complex_word_count, percentage_complex_words, fog_index, avg_words_per_sentence, personal_pronouns, avg_word_length
text=['url1.text', 'url2.text']
for i in text:
    sentiment_scores, positive_score, negative_score, polarity_score, subjectivity_score, word_count, avg_sentence_length, complex_word_count, percentage_complex_words, fog_index, avg_words_per_sentence, personal_pronouns, avg_word_length=analysis(i)
    # Print the results
    print("Positive Score:", positive_score)
    print("Negative Score:", negative_score)
    print("Polarity Score:", polarity_score)
    print("Subjectivity Score:", subjectivity_score)
    print("Avg Sentence Length:", avg_sentence_length)
    print("Percentage of Complex Words:", percentage_complex_words)
    print("FOG Index:", fog_index)
    print("Avg Number of Words per Sentence:", avg_words_per_sentence)
    print("Complex Word Count:", complex_word_count)
    print("Word Count:", word_count)
    print("Syllables per Word:", avg_syllables_per_word(text))
    print("Personal Pronouns:", personal_pronouns)
    print("Avg Word Length:", avg_word_length)


# In[32]:


import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
def analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    positive_score = sentiment_scores['pos']
    negative_score = sentiment_scores['neg']
    # Polarity and subjectivity scores
    polarity_score = sentiment_scores['compound']
    subjectivity_score = sentiment_scores['neu']

    # Text statistics
    word_count = lexicon_count(text)
    avg_sentence_length_value = avg_sentence_length(text)
    complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
    percentage_complex_words = (complex_word_count / word_count) * 100
    fog_index = 0.4 * (avg_sentence_length_value + percentage_complex_words)

    # Additional metrics
    avg_words_per_sentence = word_count / sentence_count(text)
    personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
    avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count
    return sentiment_scores, positive_score, negative_score, polarity_score, subjectivity_score, word_count, avg_sentence_length_value, complex_word_count, percentage_complex_words, fog_index, avg_words_per_sentence, personal_pronouns, avg_word_length

texts = [r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url36.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url37.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url38.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url39.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url40.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url41.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url42.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url43.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url44.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url45.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url46.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url47.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url48.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url49.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url50.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url51.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url52.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url53.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url54.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url55.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url56.txt' ]
for i, file_path in enumerate(texts):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        sentiment_scores, positive_score, negative_score, polarity_score, subjectivity_score, word_count, avg_sentence_length_value, complex_word_count, percentage_complex_words, fog_index, avg_words_per_sentence, personal_pronouns, avg_word_length = analysis(text)
        
        # Print the results
        print("File:", i+1)
        print("Positive Score:", positive_score)
        print("Negative Score:", negative_score)
        print("Polarity Score:", polarity_score)
        print("Subjectivity Score:", subjectivity_score)
        print("Avg Sentence Length:", avg_sentence_length_value)
        print("Percentage of Complex Words:", percentage_complex_words)
        print("FOG Index:", fog_index)
        print("Avg Number of Words per Sentence:", avg_words_per_sentence)
        print("Complex Word Count:", complex_word_count)
        print("Word Count:", word_count)
        print("Syllables per Word:", avg_syllables_per_word(text))
        print("Personal Pronouns:", personal_pronouns)
        print("Avg Word Length:", avg_word_length)
        print('--------------------------------------------')


# In[ ]:





# In[1]:


import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
def analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    positive_score = sentiment_scores['pos']
    negative_score = sentiment_scores['neg']
    # Polarity and subjectivity scores
    polarity_score = sentiment_scores['compound']
    subjectivity_score = sentiment_scores['neu']

    # Text statistics
    word_count = lexicon_count(text)
    avg_sentence_length_value = avg_sentence_length(text)
    complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
    percentage_complex_words = (complex_word_count / word_count) * 100
    fog_index = 0.4 * (avg_sentence_length_value + percentage_complex_words)

    # Additional metrics
    avg_words_per_sentence = word_count / sentence_count(text)
    personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
    avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count
    return sentiment_scores, positive_score, negative_score, polarity_score, subjectivity_score, word_count, avg_sentence_length_value, complex_word_count, percentage_complex_words, fog_index, avg_words_per_sentence, personal_pronouns, avg_word_length

texts=[r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url57.1.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url58.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url59.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url60.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url61.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url62.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url63.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url64.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url65.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url66.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url67.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url68.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url69.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url70.txt']
for i, file_path in enumerate(texts):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        sentiment_scores, positive_score, negative_score, polarity_score, subjectivity_score, word_count, avg_sentence_length_value, complex_word_count, percentage_complex_words, fog_index, avg_words_per_sentence, personal_pronouns, avg_word_length = analysis(text)
        
        # Print the results
        print("File:", i+1)
        print("Positive Score:", positive_score)
        print("Negative Score:", negative_score)
        print("Polarity Score:", polarity_score)
        print("Subjectivity Score:", subjectivity_score)
        print("Avg Sentence Length:", avg_sentence_length_value)
        print("Percentage of Complex Words:", percentage_complex_words)
        print("FOG Index:", fog_index)
        print("Avg Number of Words per Sentence:", avg_words_per_sentence)
        print("Complex Word Count:", complex_word_count)
        print("Word Count:", word_count)
        print("Syllables per Word:", avg_syllables_per_word(text))
        print("Personal Pronouns:", personal_pronouns)
        print("Avg Word Length:", avg_word_length)
        print('--------------------------------------------')


# In[5]:


import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
def analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    positive_score = sentiment_scores['pos']
    negative_score = sentiment_scores['neg']
    # Polarity and subjectivity scores
    polarity_score = sentiment_scores['compound']
    subjectivity_score = sentiment_scores['neu']

    # Text statistics
    word_count = lexicon_count(text)
    avg_sentence_length_value = avg_sentence_length(text)
    complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
    percentage_complex_words = (complex_word_count / word_count) * 100
    fog_index = 0.4 * (avg_sentence_length_value + percentage_complex_words)

    # Additional metrics
    avg_words_per_sentence = word_count / sentence_count(text)
    personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
    avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count
    return sentiment_scores, positive_score, negative_score, polarity_score, subjectivity_score, word_count, avg_sentence_length_value, complex_word_count, percentage_complex_words, fog_index, avg_words_per_sentence, personal_pronouns, avg_word_length

texts=[r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url71.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url72.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url73.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url74.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url75.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url76.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url77.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url78.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url79.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url80.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url81.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url82.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url83.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url84.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url85.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url86.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url87.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url88.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url89.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url90.txt']
for i, file_path in enumerate(texts):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        sentiment_scores, positive_score, negative_score, polarity_score, subjectivity_score, word_count, avg_sentence_length_value, complex_word_count, percentage_complex_words, fog_index, avg_words_per_sentence, personal_pronouns, avg_word_length = analysis(text)
        
        # Print the results
        print("File:", i+1)
        print("Positive Score:", positive_score)
        print("Negative Score:", negative_score)
        print("Polarity Score:", polarity_score)
        print("Subjectivity Score:", subjectivity_score)
        print("Avg Sentence Length:", avg_sentence_length_value)
        print("Percentage of Complex Words:", percentage_complex_words)
        print("FOG Index:", fog_index)
        print("Avg Number of Words per Sentence:", avg_words_per_sentence)
        print("Complex Word Count:", complex_word_count)
        print("Word Count:", word_count)
        print("Syllables per Word:", avg_syllables_per_word(text))
        print("Personal Pronouns:", personal_pronouns)
        print("Avg Word Length:", avg_word_length)
        print('--------------------------------------------')


# In[6]:


import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_kincaid_grade, flesch_reading_ease, syllable_count, lexicon_count, sentence_count, avg_sentence_length, avg_syllables_per_word

# Text for analysis
def analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    positive_score = sentiment_scores['pos']
    negative_score = sentiment_scores['neg']
    # Polarity and subjectivity scores
    polarity_score = sentiment_scores['compound']
    subjectivity_score = sentiment_scores['neu']

    # Text statistics
    word_count = lexicon_count(text)
    avg_sentence_length_value = avg_sentence_length(text)
    complex_word_count = len([word for word in nltk.word_tokenize(text) if syllable_count(word) > 2])
    percentage_complex_words = (complex_word_count / word_count) * 100
    fog_index = 0.4 * (avg_sentence_length_value + percentage_complex_words)

    # Additional metrics
    avg_words_per_sentence = word_count / sentence_count(text)
    personal_pronouns = len([word for word in nltk.word_tokenize(text) if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
    avg_word_length = sum(len(word) for word in nltk.word_tokenize(text)) / word_count
    return sentiment_scores, positive_score, negative_score, polarity_score, subjectivity_score, word_count, avg_sentence_length_value, complex_word_count, percentage_complex_words, fog_index, avg_words_per_sentence, personal_pronouns, avg_word_length

texts=[r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url101.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url102.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url103.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url104.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url105.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url106.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url107.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url108.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url109.txt', r'C:\Users\areeb\OneDrive\Desktop\blackcoffer\url110.txt']
for i, file_path in enumerate(texts):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        sentiment_scores, positive_score, negative_score, polarity_score, subjectivity_score, word_count, avg_sentence_length_value, complex_word_count, percentage_complex_words, fog_index, avg_words_per_sentence, personal_pronouns, avg_word_length = analysis(text)
        
        # Print the results
        print("File:", i+1)
        print("Positive Score:", positive_score)
        print("Negative Score:", negative_score)
        print("Polarity Score:", polarity_score)
        print("Subjectivity Score:", subjectivity_score)
        print("Avg Sentence Length:", avg_sentence_length_value)
        print("Percentage of Complex Words:", percentage_complex_words)
        print("FOG Index:", fog_index)
        print("Avg Number of Words per Sentence:", avg_words_per_sentence)
        print("Complex Word Count:", complex_word_count)
        print("Word Count:", word_count)
        print("Syllables per Word:", avg_syllables_per_word(text))
        print("Personal Pronouns:", personal_pronouns)
        print("Avg Word Length:", avg_word_length)
        print('--------------------------------------------')


# In[ ]:




