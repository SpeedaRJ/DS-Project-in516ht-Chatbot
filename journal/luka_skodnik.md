# Luka Škodnik's Data Science Project Competition journal

## March 2020 (17h30min)

* **1st** (1h): Meeting with everyone - quick conversation about the data we are working with, establishing expectations, deciding on preliminary goals, setting up feature meetings and weekly schedule.
* **6th** (2h): Working with my coworker - we looked at the agreed upon data, outlined important and numerical data, found some extraction tools for images, tables, text etc. We tested some, but not all, of them.
* **7th** (30min): Looking at some tools for text extraction
* **8th** (30min): Meeting with NLB advisors
* **9th** (1h):
  * (30min) Reading papers
  * (30min) Meeting with professor
* **11th** (3h): Watching videos and reading papers about attention, transformers, large language models, question answering models
* **12th** (4h):
  * (2h30min) Continue reading papers, gathering info what datasets are used in question answering models, which question answering models are used, what are their differences
  * (1h30min) Reading about the capabilities of haystack (deepset end-to-end framework) and trying to get it to work on my machine
* **13th** (2h): managed to install haystack, trying out it's features
* **15th** (1h): Meeting with NLB and In516ht advisors
* **22th** (30min): Meeting with NLB advisor
* **27th** (1h30min): Reading some papers and started organizing the intermediate report
* **30th** (30min): Meeting with NLB and In516ht advisors

## April 2020 (36h)

* **1st** (2h): Looking through possible models to fine tune on our data and how we should go about it.
* **4th** (6h): Working with my coworker - getting a baseline comparison of predictions from a model without fine tunning. Working on the environment, and preparing a feature plan for development and testing of the main model.
* **5th** (8h): Working with my coworker - original fine tuning, setting up a Sequence Extractive model, planning the next steps, meeting with the professor in the meantime.
* **11th** (8h): Working with my coworker - fine tuning, baseline evaluation, planning the next steps.
* **12th** (10h): Working with my coworker - fine tuning, baseline evaluation, planning the next steps, meeting with the professor and meeting with NLB advisors.
* **13th** (2h): Tweaking, modifying and proofreading the intermediate report, looking at some possibilities for the generative approach.

## May 2020 (91h30min)

* **5th** (5h): Setting up another environment with cudatoolkit, initial T5 testing and trying to set up fine tunning
* **6th** (8h): Setting up fine tunning with the example script with no success, trying to find a solution
* **7th** (3h): Still trying to find a solution for fine tunning, looking at already trained DPR models
* **8th** (6h): Managed to get the fine tunning script to work, debugging problems with padding tokens, trying to find a solution. Trying training with different parameters. Looking at other generative model possibilities (LLaMA).
* **9th** (7h): Working with my coworker - updating the environment, testing other generative models, looking at how to set up DPR and the entire pipeline.
* **10th** (6h): Working with my coworker - Meeting with NLB - talking about the final data (we should get it by the end of the week), planning the final report, setting up pipeline, trying to debug the base t5 model without success.
* **11th** (3h): Meeting with professor, trying to debug the base t5 and get predictions from LLaMA model without success.
* **12th** (3h): Still trying to debug the base t5 model, trying to get predictions from LLaMA model. Realizing that for efficiently running LLaMA we need the environment on databricks set up (it runs too slowly on local machine) - we probably wont have enough time to test this thoroughly and implement it to the end.
* **15th** (6h): Starting to reorganize the report for the final submission. Looking at the new data (handwritten one) and training models that we will need for the results in report. Realizing that the "bug" with t5 base size model probably is not a bug as with a different provided context we get results as expected - just weird behavior. (why did i spend so much time on this and i'm still not sure 100% (╯°□°）╯︵ )
* **16th** (6h): Continuing work from yesterday - new data, models, results, report.
* **17th** (8h): Meeting with NLB advisor, working with my coworker - continuing work from yesterday - new data, models, results, report.
* **18th** (8h): Working with my coworker - continuing work from yesterday - new data, models, results, report.
* **19th** (1h): Realizing we have a small bug and fixing it
* **20th** (3h): Updating evaluation, models
* **21st** (1h30min): Cleaning up things in interim report, creating table templates for final report, rereading related work and methodology to see how much we need to change
* **22nd** (9h): Working with my coworker - qualitative evaluation, writing report, finalizing the project
* **23rd** (8h): Qualitative evaluation - looking through examples to include in report, writing the report to send for review before the final submission

## June 2020 ([total hours for June])

...

## Total: 145h