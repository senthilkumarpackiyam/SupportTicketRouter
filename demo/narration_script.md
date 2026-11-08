# Demo video: narration script

Voice: macOS text-to-speech, **Aman (English (India))** (Indian English, male). Built by [`demo/build/make_demo.py`](build/make_demo.py).

Length: 5 min 5 s · subtitles: [`SupportTicketRouter_demo.srt`](SupportTicketRouter_demo.srt)

## 00:00:00 · title (slide)

> Hello, and welcome to this demo of the Support Ticket Router.
> In this project, a small open model, Qwen3 1.7 billion, is fine-tuned with LoRA, using LLaMA Factory, to route internal IT support tickets to the right team.
> This is the Week 5 project of The Gen Academy.

## 00:00:17 · business (slide)

> An IT help desk receives tickets by email, by Slack, and through a help portal.
> Today, a person reads every single ticket just to decide which queue it belongs to.
> The router reads the ticket first, and answers with exactly one of seven labels.
> It never writes a reply. It only routes, and the rest of the tooling takes over.
> A small fine-tuned model is much cheaper and faster for this than calling a frontier model for every ticket.

## 00:00:45 · data (slide)

> The dataset has 585 labelled tickets.
> We made a stratified 80/20 split: 468 tickets for training, and 117 held out for validation, which the model never sees.
> The training tickets are converted to ShareGPT chat format, and registered in LLaMA Factory as the support tickets dataset.
> Before training, we checked for leakage between the splits, and verified exactly which tokens the model learns from.

## 00:01:14 · llamaboard setup (screen recording `llamaboard_train_session.mp4` 0–22 s at 0.67× speed)

> Training happens in LLaMA Board, the web interface of LLaMA Factory.
> Here, a Playwright script drives the browser, so every click is reproducible.
> It selects the Qwen3 1.7 billion base model, LoRA as the fine-tuning method, and the support tickets dataset.
> It previews the data and the exact training command, and then presses Start.
> We kept LLaMA Board's defaults: learning rate 5e-5, three epochs, LoRA rank 8, and an effective batch size of 16.

## 00:01:46 · training timelapse (screen recording `llamaboard_train_session.mp4` 200–1498 s at 77.39× speed)

> Instead of a Colab T4, training ran locally on an Apple M2 laptop GPU.
> Here is the first epoch as a time-lapse. The loss curve in LLaMA Board drops quickly.
> Only 8.7 million parameters are trained. That is half a percent of the model.

## 00:02:04 · loss (slide)

> The full run took 90 steps and about 69 minutes.
> The loss falls from 7.16 to 0.19, a drop of 97%, and it stays flat during the third epoch.
> So the model has converged, and there was no need to retrain.

## 00:02:21 · llamaboard chat (screen recording `llamaboard_chat_session.mp4` 0–40 s at 1.50× speed)

> Back in LLaMA Board, the Chat tab loads the base model together with the trained adapter.
> We send tickets in the same format used for training, and the model answers with a single label.
> The first ticket, a new user account request, comes back as Support general. We will come back to that.
> The Outlook and Teams problem goes to O365, and retiring a server goes to EOL. Both are correct.

## 00:02:48 · merge smoke (slide)

> Next, the adapter is merged into the base weights, to give one standalone model.
> We checked that the merged model gives the same output as the adapter.
> On the course's five smoke test tickets, it gets four right.
> The miss is interesting. In the training data, new user account requests are labelled Fileservice nine times, Support general three times, and Active Directory only twice.
> So the model simply learned the convention in the data.

## 00:03:15 · results (slide)

> Now the real test: all 117 held-out tickets.
> The base model, given the seven labels as multiple choice from A to G, reaches only 24.8%. It mostly just picks option A.
> The fine-tuned model reaches 72.6% accuracy. That is a gain of 48 points, and the macro F1 score rises from 0.08 to 0.68.

## 00:03:41 · confusion (slide)

> The confusion matrix shows where the model works, and where it does not.
> EOL is perfect, 12 out of 12, and Fileservice is strong.
> The weak spot is Active Directory: only 2 of 9 tickets are caught. Most errors drift to the catch-all Support general queue.
> Along the way, we also found and fixed a bug in the course notebook. Its classification report was called without the labels argument, so every row had the wrong class name.

## 00:04:09 · confidence (slide)

> The model's confidence score is genuinely useful.
> If we route automatically only when the confidence is at least 0.8, 44% of tickets are handled at 86% accuracy, and the rest go to a person.
> Each prediction takes about 280 milliseconds on the laptop GPU.

## 00:04:28 · context (slide)

> For honest context, a simple TF-IDF and logistic regression model reaches 69% on the same split.
> With only 117 test tickets, the fine-tuned model's lead is within the margin of error.
> The next steps are clear: agree on a labelling rule for account requests and relabel them, auto-route only confident predictions, and try a second training run with a higher learning rate.

## 00:04:53 · close (slide)

> Everything is in the GitHub repository: the code, the trained adapter, the executed notebook, and a write-up for every phase.
> Thank you for watching!

