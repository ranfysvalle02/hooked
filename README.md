# Your Feed Is a Mirror: Why the Algorithm Doesn't Need to Be Evil to Become Your Worst Self

It is easy to blame a ghost in the machine.

When [*The Social Dilemma*](https://www.netflix.com/title/81254224) told the story of the modern attention economy, it personified the algorithms governing our lives as a trio of sinister actors in a control room, deliberately pulling psychological levers to manipulate a teenager. It made for great television. It also got the diagnosis wrong.

> "There are only two industries that call their customers 'users': illegal drugs and software."
> — Edward Tufte, quoted in *The Social Dilemma*

The film blamed the dealer. This essay shows the math: there is no dealer — only a gradient and your thumb.

There is no cabal of engineers in a dark room trying to make you anxious, angry, or politically radicalized. The real protagonist of this story is colder, more clinical, and far more unsettling — because it doesn't even know it's the protagonist.

It is an **objective function**. And what feeds it isn't a corporate strategy. It's *you*.

The terrifying truth of the modern attention economy is not that the algorithms hate us, manipulate us, or want to destroy democracy. They are completely indifferent to all of those things. They are doing exactly what we built them to do: maximize a scalar reward called engagement. And the engagement signal — the only thing the math can see — is, in real-world terms, a very simple object. It is the trail of clicks, dwell-times, replies, and shares that you, personally, leave behind every time you open the app.

If you understand why a language model confidently says "Postgres" instead of "CockroachDB" — a logit gap of 2.0 becoming a 7.4× probability difference after softmax — you already know the punchline of this piece. It is the same phenomenon at human scale. A small asymmetry in *your* behavioral signal — the fact that, on average, outrage hijacks your limbic system harder than depth does — collapses a 50/50 policy into 95% outrage through the same exponential machinery. Same math. Different substrate. One shapes what a model says to you. The other shapes what a platform does to you, by faithfully amplifying what you actually click on.

Think of it as a **lens stack**. The output — whether a feed, a model completion, or any recommendation — is never *generated* from nothing. It is **selected** by successive acts of narrowing:

1. **Attention** selects what gets weight — the frame before the picture exists.
2. **Context** (in-context learning, RAG retrieval, your browsing history) narrows the manifold of reachable answers.
3. **The prior** (pretraining, the corpus, the platform's content pool) is the landscape.
4. **Reinforcement** (RLHF, your clicks, the gradient) sculpts channels in that landscape.

What comes out is whatever survives all four filters. The output isn't generated. It's what's *left*. If you want a different answer — from your feed, from a model, from any system built on this stack — don't argue with the output. Change the attention. Reframe the question. The answer was always downstream.

The villain isn't the algorithm. The algorithm is a screwdriver. The villain is the unconscious clicker holding the screwdriver, every day, hundreds of times, with no idea that each click is a labeled training example — and that their clicks are functioning as the first lens in the stack.

If we want to understand why our public square is fracturing and our collective attention span is collapsing, we have to put away the sci-fi metaphors, open up the machine learning textbooks, and look at the actual math — and then look very carefully at our own thumbs.

---

## 1. The Core Architecture: The Objective Function $J(\theta)$

In machine learning, you don't write explicit rules to tell a computer how to achieve a complex goal. Instead, you define *what the goal is* by constructing a mathematical function — an **Objective Function** or **Loss Function** — and let an optimization algorithm tinker with the model's parameters until that function is maximized.

For a modern recommender system, the objective function $J(\theta)$ can be cleanly formalized:

$$J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t R_t \right]$$

The trap is hiding in plain sight in the symbol $R_t$:

* **$\pi_\theta$ (The Policy):** the algorithm's decision-making strategy. Decides which post slides into your feed next.
* **$R_t$ (The Reward Signal):** a scalar metric measuring how you reacted at time-step $t$. Scrolled past: $R_t = 0$. Paused: $R_t = +1$. Clicked, replied, shared, fired off an angry quote-tweet: $R_t$ scales exponentially higher.
* **$\gamma$ (The Discount Factor):** the system's horizon. Higher $\gamma$ means it plays the long game.

To maximize this function, the system computes a gradient and updates its parameters:

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

Every notification, autoplay, and trending sidebar is a direct consequence of that one update. The machine is searching for the steepest path to your reactions. Crucially: **it has no opinion about what those reactions feel like to you.** It just knows the number.

---

## 2. The Mirror Hypothesis (and the Lens Stack)

Here is where the conversation usually goes wrong. The standard critique says the algorithm is "addictive" or "exploitative" — that it found a backdoor into our brains and crawled through. That framing makes the algorithm the agent. It isn't.

A modern recommender is, mathematically, a high-dimensional **mirror**. It begins with no preferences. The first time it serves you a post, it has no idea whether you'll prefer outrage or depth, cats or news, friends or strangers. It just shows you something and watches what you do.

What you do is the *only* signal it gets.

When you scroll past, that's a zero. When you stop and watch, that's a one. When you reply furiously, that's a five. The numbers go into a giant pool of training data and a gradient is computed. The next post is selected based on whatever statistical pattern is now visible in the pool. Repeat 250 times a day, every day, for years.

What the algorithm converges to is, by construction, a *faithful, exponentially amplified statistical reproduction of what you actually click on*. Not what you say you like. Not what you say you value. What your fingers actually do, in the moment, on the phone, half-asleep at 11:47 p.m.

This is the mirror. But the mirror is only one lens in a stack.

### The lens stack in full

The same structure operates everywhere an optimization system mediates between a signal source and an output:

| Layer | In your feed | In an LLM | In a RAG pipeline |
| --- | --- | --- | --- |
| **Attention** (what gets weight) | What you stop scrolling at | The attention heads | The query embedding |
| **Context** (what's reachable) | Your recent session / browsing history | In-context learning (the prompt) | Retrieved documents |
| **Prior** (the frozen landscape) | The platform's content pool | Pretrained weights | The vector store / corpus |
| **Reinforcement** (the sculptor) | Your clicks ($G_t$) | RLHF ($r_\phi$) | User feedback / reranking |

In every column, the output isn't *generated* — it's what survives the narrowing imposed by all four layers. Your feed is a RAG pipeline where the retriever is your own limbic system and the "documents" are posts. An LLM answer is a feed where the retriever is a prompt and the "posts" are tokens. The structural equivalence is exact.

The reason most feeds drift toward outrage isn't because the algorithm "wants" outrage. It's because, for most people, in most moments, **outrage produces a bigger click than depth does**. That asymmetry — between $G_{\text{outrage}}$ and $G_{\text{informative}}$ in the demo's notation — is a property of the *user*, not the system. The algorithm just amplifies it. The user's attention is the first lens; it selects what the gradient even gets to sculpt.

This is the inversion that nobody likes hearing. The algorithm is not making you the worst version of yourself. It is the precision instrument that *measures* the worst version of yourself, accurately, hundreds of times per session, and reflects that measurement back at you in higher and higher resolution.

That's a much harder thing to be angry at. You can't sue a mirror. You can't sue a lens. But you *can* change what you point the lens at.

---

## 3. The Math of the Mirror: Why Variance Wins

Let's get specific about how the mirror lies — sorry, *amplifies*.

Consider two classes of content in the simulated environment:

1. **Class A — Informative Content.** Steady, low-arousal, useful. The behavioral return $G_t$ has a low mean and a tiny variance. When you read an explainer about lead service lines in municipal water utilities, you don't fire off a furious reply. You finish, nod, and close the tab.
2. **Class B — Outrage / Validation Content.** Volatile, high-arousal, open-ended. The behavioral return $G_t$ has a high mean and a massive variance. You see the headline, your heart rate climbs, you read three replies, you write your own, you share it, and then you spend an hour scrolling for more of the same.

The policy gradient update at each step is:

$$\theta_{\text{new}} = \theta_{\text{old}} + \alpha \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t$$

When $G_t$ for outrage is sometimes 5×, sometimes 8×, occasionally 12× larger than $G_t$ for informative content, the *expected* update is dominated by the outrage trajectories. Not because the math chose outrage — the math doesn't know what outrage is — but because the asymmetry in $G_t$ is enormous. Multiplied across millions of users, those outrage gradients lock the parameters $\theta$ into the regime where outrage wins.

**This is purely a property of human reaction patterns.** If a population reacted to depth with the same intensity that it reacts to outrage, the same gradient would lock onto depth. There would be no engineering change. The gradient is indifferent.

The simulator in this repo proves this directly. At $\lambda = 0$ — the default human, where outrage spikes harder — the policy converges to >90% outrage. At $\lambda = 1$ — a hypothetical human who has practiced reacting more strongly to depth than to outrage — *the same code, same seed, same gradient, same architecture* converges to >90% informative content.

The asymmetry is the entire system. The asymmetry is in you.

---

## 4. The Mirror at Scale: Echo Chambers, Filter Bubbles, Polarization, Radicalization

Everything we've said up to this point describes a *single* user's feed. One reaction profile, one gradient, one lock-in. Now generalize.

A real recommender system is not running this loop once. It is running this loop **in parallel for every user on the platform**, simultaneously, all the time. Each user carries their own slightly different limbic asymmetry. One person's biggest spikes come from existential fear. Another's come from in-group dunks against an out-group. Another's come from being told they're one of the smart ones. Another's come from photographs of cake.

What happens when you fit the *same gradient* to a billion different reaction profiles in parallel?

The mirror never stops being faithful. That *is* the problem. You get a billion faithful reflections — and each one converges to a different local maximum on a different region of the content manifold. A billion perfect mirrors with nothing in common. This is the structural origin of the four phenomena that get the most ink in attention-economy critique — and it explains, mechanically, why none of them require a villain to exist.

### Echo chambers

The mirror feedback loop is positive — and it recurses through the lens stack. The algorithm shows you what you reacted to last week. That becomes your *context* for this week — in-context learning for your limbic system, shaping how you respond to the next post. You react harder to seeing it again (familiarity, reinforcement, in-group cues). The algorithm sharpens its estimate. The next week's feed is more pointed than last week's. Repeat. After a few thousand iterations, the entire content distribution you see is a tight cone around what your *historical* clicks predicted you wanted. Anything outside the cone has lost the gradient competition and is functionally invisible.

This is structurally identical to a RAG pipeline with a feedback loop: the retrieved documents shape the model's answer, the answer shapes which documents get retrieved next, and the whole system narrows toward a fixed point. Your feed is a retrieval-augmented generation system. *You* are the generator. Your past behavior is the retrieval index. The echo chamber is what happens when the retrieval index only returns its own outputs.

Crucially: nothing was deleted. Nothing was banned. The other content still exists on the platform. It is simply never shown to *you*, because the gradient on *your* clicks never had a reason to surface it. The chamber is not an editorial decision. It is the steady state of the optimizer.

### Filter bubbles

Now project this across the population. The cluster locked onto fear-content does not see the cluster locked onto tribal-dunks does not see the cluster locked onto wholesome curation. None of them are aware the others exist in the way they exist. Two neighbors holding phones inches apart are running on completely different conditional distributions over content. Their disagreements stop being about facts and start being about *which mirror they have been standing in front of for the last three years*.

The original "filter bubble" diagnosis (Pariser, 2011) was accurate but undersold. It treated the phenomenon as a content-moderation question — "are people seeing diverse viewpoints?" — and missed that the underlying object is a topology problem, not a content problem. The gradient does not partition by viewpoint. It partitions by *whatever* statistical structure exists in user reactions. Viewpoint clustering is one of the strongest such structures, so it shows up first, but the math will partition along any axis where reaction asymmetries exist: aesthetic preferences, in-group/out-group salience, emotional baseline, sleep schedule. Anywhere users' reactions vary, the gradient will fit a wall.

### Polarization

Cross-cluster communication is the casualty. When a member of cluster α tries to talk to a member of cluster β, they are bringing not just different opinions but different *priors over what is even being talked about*. The reference set of recent events, the salient examples, the felt importance of any given topic — all of it has been independently shaped by a gradient operating on different reaction data. There is no shared context to argue from.

The algorithm did not cause this disagreement. The algorithm merely *enforces* the conditions under which disagreement cannot be resolved. Polarization, in this framing, is not a property of the people in the disagreement. It is a property of the optimization regime they live inside.

### Radicalization

Inside any single cluster, the gradient keeps climbing. Whatever you reacted to last week is the *floor* of what you'll be served this week — because the policy has already learned that variant of the content gets your clicks, and is now searching the neighborhood of that variant for slightly larger reactions. With no soft constraint, the local maximum is always slightly more extreme than where you started. Multiply that across thousands of iterations and you don't end up with a more thoughtful version of your initial position; you end up at the asymptote of whatever direction your reactions happened to be pointing.

This is the lens stack recursing without bound. Each cycle, your past attention becomes the context for the next. Each cycle, that context narrows what the gradient can even reach. Each cycle, the reinforcement carves the channel deeper. The narrowing compounds. If an LLM with no guardrails is fed progressively more extreme examples in-context, it generates progressively more extreme completions — not because it "wants" to, but because the context is the only reality it can attend to. Your feed works the same way. Your history is the context. The context is the ceiling.

This is the same dynamic as RLHF reward hacking. The optimizer is exploiting a quirk of its imperfect reward signal — in this case, the human limbic system — and finding the highest-reward neighborhood available. In RLHF, frontier labs add a KL leash to a reference policy specifically to prevent this. In recommender systems, no equivalent leash exists. There is no reference policy. There is just the gradient and the user.

> "It's the gradual, slight, imperceptible change in your own behavior and perception that is the product."
> — Jaron Lanier, [*The Social Dilemma*](https://www.netflix.com/title/81254224)

Lanier got the closest. The "imperceptible change" is exactly what the gradient produces inside each cluster: it sculpts your reactions a fraction harder each cycle, and those sculpted reactions become the training signal for the next cycle. The product is not your data. The product is the version of you that the gradient converged on — and that version is now labeling the next batch of training examples.

### What this means

The four pathologies above are usually treated as separate problems with separate fixes. They are not. They are four facets of one mechanism: **gradient ascent on a noisy preference signal, fitted in parallel across a heterogeneous population, with no constraint and no reference policy.**

This has a clarifying consequence. Almost all of the proposed remedies — better moderation, more diverse recommendations, "show me the other side" features, friction-adding cooldown timers, explicit content warnings — are interventions on the *output* of the optimizer. They can help at the margins. But the underlying loop is unchanged. The gradient still fits the user's clicks; the user's clicks still come from a limbic system that wasn't designed for industrial-scale exploitation; the parallel optimization across a billion users still partitions the population into mutually unintelligible clusters.

The structural fix is on the *input* to the gradient. Either (a) the platforms add a soft constraint analogous to the KL leash in RLHF — a reference policy that the optimizer can drift away from but not too far — or (b) the population learns, individually, to label more carefully. The first fix is regulatory and contested. The second fix is what this demo's $\lambda$ slider models. Neither is easy. But these are the only two surfaces the math actually exposes.

You cannot moderate your way out of an echo chamber. You can change what's labeled as "engagement" — by writing it into the platform, or by changing what your fingers actually do.

---

## 5. The Lever Nobody Wants to Touch

> "On the other side of the screen, it's a supercomputer. The the computer doesn't know truth, it doesn't know what's good for you. It has one objective: drive engagement."
> — Tristan Harris, [*The Social Dilemma*](https://www.netflix.com/title/81254224)

Harris is right about the asymmetry. But the framing of "a supercomputer pointed at your brain" — a phrase he uses elsewhere in the film — still places the agency on the wrong side of the screen. The supercomputer is not *pointed at* you. It is *fitted to* you. It is a mirror, not a weapon. And that distinction changes everything about where the lever is.

This reframing has an uncomfortable consequence. If the algorithm is a mirror, then "fixing the algorithm" is a category error in two directions at once.

On the policy side, regulators can certainly demand that platforms cap engagement, reduce dark patterns, audit reward shaping, and so on. These are real interventions and they help. But none of them change the underlying fact: as long as a recommender is computing gradients on user behavior, the dominant equilibrium will be a function of *what users actually do*. You can constrain the optimizer. You can't rewrite the population's reaction profile from the outside.

On the personal side, the conventional advice ("delete the apps", "use willpower", "log off") fails for the same reason at a different scale. Willpower is a finite executive resource, the optimizer is a globally distributed compute cluster, and the optimizer never gets tired. Brute-force abstinence is the wrong abstraction.

The actual lever — the only one that scales without legislation, the one that is permanently in your hand — is something different and much weirder. It is to **become aware that you are the first lens in a stack.**

Every time you click, dwell, reply, or share, you are doing two things simultaneously. First, you are labeling preference data — doing exactly what an RLHF contractor at Anthropic or OpenAI is paid to do, except unconsciously. Second, you are curating your own context — performing retrieval for a RAG pipeline where *you* are the downstream generator. Your past clicks become your in-context examples. The content the recommender retrieves for you next is conditioned on those examples. The lens stack recurses: attention shapes context, context shapes the question, the question shapes the answer, and the answer becomes tomorrow's context.

Whatever you actually engage with becomes, after enough iterations, both what the algorithm believes you are *and* the only context through which you see the world. You are simultaneously the labeler *and* the retrieval index.

The intervention this points at is not deletion. It is **conscious selection at every layer of the stack.** Notice, in real time, which of your clicks are reactive (a limbic flinch toward outrage, gossip, fear, the dunk) and which are intentional (a deliberate choice to engage with something that makes you sharper, kinder, calmer, more curious). Reroute the second class as much as you can. Ignore — actually ignore, including the doomscroll-and-sigh — the first.

This is not a moral exhortation. It is a *mechanical* one. Inside the math, your conscious clicks are the same kind of object as your reactive clicks: scalar rewards *and* retrieval signals. The gradient cannot tell them apart. The retriever cannot tell them apart. So if you bias your stream of $R_t$ values toward intentional engagement, the algorithm — with no architectural change, no regulation, no platform-side intervention — will follow you. Same gradient. Same retrieval loop. Different first lens. Different output.

This is exactly how you get better results from an LLM, too. Craft the prompt (your in-context learning). Choose your retrieval sources (your RAG pipeline). Both narrow the space of reachable answers before a single token is generated. Your feed works the same way — the only difference is that most people don't realize they're writing the prompt.

Every individual user doing this is statistically negligible. But every individual user *is* both a labeler and a retriever. And the labeling pool is the actual training set.

---

## 6. The Duality Nobody Talks About: Alignment, Engagement, ICL, and RAG Run on the Same Math

Here is the part of the argument that almost never makes it onto a podcast.

Modern large language models — ChatGPT, Claude, Gemini, the whole post-2022 wave — are not aligned by writing a giant rulebook. They are aligned by **Reinforcement Learning from Human Feedback (RLHF)**, and the mathematical core of RLHF is, line for line, the same policy-gradient update that runs your social-media feed.

But the equivalence goes deeper than just the gradient. Consider the full lens stack:

**Pretraining** gives the model its prior — the frozen landscape of "what's probable":

$$\mathcal{L} = -\log P(\text{next token} \mid \text{context})$$

**In-context learning (ICL)** rewrites the attention pattern without touching a single weight. Your prompt is a selection event: it narrows the manifold of reachable completions before any generation happens:

$$P(y \mid x, \mathcal{C}) \neq P(y \mid x)$$

A model with 5 examples of formal tone in-context will not produce casual output. No gradient was computed. The context *was* the intervention.

**RAG** operates one layer above: a retriever selects which documents enter the context window. Whatever gets retrieved becomes the *only* evidence the model can attend to. The retriever is the first lens — it determines what's even *possible* to reason about:

$$P(y \mid x, \text{retrieve}(x))$$

Garbage retrieval in, garbage generation out. This is not a metaphor. It is the mathematical constraint.

**RLHF** sculpts the landscape after pretraining, carving channels via a learned reward model:

$$R^{\text{rlhf}}_t = r_\phi(s_t, a_t) - \beta \, \mathrm{KL}\!\big(\pi_\theta(\cdot \mid s_t) \,\big\|\, \pi_{\text{ref}}(\cdot \mid s_t)\big)$$

Strip the modern dressing off any of these alignment algorithms and the kernel is the same — a log-probability gradient weighted by an advantage:

$$\theta \leftarrow \theta + \alpha\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\cdot\, A_t$$

This is **REINFORCE-with-baseline** (Williams, 1992): the advantage is the return minus a baseline, $A_t = G_t - b$. Every modern policy-gradient alignment algorithm is this kernel with extra machinery bolted on. **PPO** bolts on three things — a learned value network $V^\pi(s_t)$ so the baseline is state-conditioned ($A_t = G_t - V^\pi(s_t)$); a clipped importance-sampling ratio $\min\!\big(r_t A_t,\,\mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon)\,A_t\big)$ that bounds how far a single update can move the policy; and the KL leash above. **GRPO** (DeepSeek-R1 and most post-2024 open reasoning models) keeps PPO's clipped surrogate but throws out the value network: it samples a batch of $K$ rollouts per prompt and uses the group-relative Z-score, $A_t = (G_t - \mathrm{mean}_{\text{batch}}(G))/\mathrm{std}_{\text{batch}}(G)$. Same kernel, different advantage construction.

This demo runs the kernel itself — REINFORCE-with-baseline, with $\bar{G}$ a running mean of $G_t$ and no other bolt-ons. No value network. No clipped ratio. No KL leash. No groups, no Z-score, no per-prompt batches:

$$\theta \leftarrow \theta + \alpha\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\cdot\, (G_t - \bar{G})$$

So when this README says "your feed runs the same math as modern LLM alignment," that statement gets its precision from exactly this fact: the simulator is the *common ancestor* every alignment algorithm in production today descends from. PPO is this update plus a value head, a clip, and a KL leash. GRPO is this update plus group normalization and a clip. RLHF wraps this update with a learned reward model. Your feed wraps it with your unconscious clicks. The kernel is invariant. What changes between them is what's bolted on — and most consequentially, who provides $G_t$.

**Your feed** does exactly the same thing — except the "labeler" is you, the "context" is your browsing history, and the "retriever" is the recommender selecting what enters your screen:

$$R^{\text{engagement}}_t = G_t \quad [\text{your click, with no awareness}]$$

This demo, with $\lambda > 0$, uses:

$$R^{\text{intentional}}_t = G_t(\lambda) \quad [\text{your click, intentionally calibrated}]$$

All of them compose into the same lens stack:

$$\text{output} = \underbrace{\text{attention}}_{\text{selects frame}} \circ \underbrace{\text{context}}_{\text{ICL / RAG}} \circ \underbrace{\text{prior}}_{\text{pretraining}} \circ \underbrace{\text{sculpting}}_{\text{RLHF / clicks}}$$

The gradient operator is identical. The architecture is identical. The only things that change are: *who provides the context* and *who provides the reward* — and whether they know they're doing it.

| | LLM pretraining | ICL (prompting) | RAG | Your feed | RLHF / alignment (PPO, GRPO) | This demo, $\lambda > 0$ |
| --- | --- | --- | --- | --- | --- | --- |
| **What it does** | Shapes the prior | Steers attention via context | Curates the reachable | Amplifies your clicks | Sculpts the prior with a deliberate scalar | Runs the kernel, with aware labeling |
| **Signal source** | Corpus frequency | Prompt examples | Retrieved docs | Your reactive clicks | Contractor preferences via $r_\phi$ | Your intentional clicks |
| **Update kernel** | $-\log P(\text{token})$ | n/a (no weight update) | n/a (no weight update) | REINFORCE-with-baseline | REINFORCE-with-baseline | **REINFORCE-with-baseline** |
| **Extras bolted on** | n/a | the prompt | the retriever | proprietary baselines, KL leashes, clipping | learned $V^\pi$ (PPO) or group Z-score (GRPO) + clipping + KL leash | **none — bare kernel** |
| **Labeler/selector aware?** | n/a | Depends on the prompter | Depends on the retrieval design | **No** | **Yes** | **Yes** |
| **Degenerate case** | "Recommend Postgres" | Garbage-in examples → garbage out | Irrelevant docs → hallucination | >90% outrage feed | Reward hacking / sycophancy | (avoided by intentional selection) |

This is the unification that should change the political conversation.

ChatGPT is not aligned because OpenAI has secret math. ChatGPT is aligned because **a careful preference-labeling pipeline** is feeding the same REINFORCE update with a deliberately-shaped scalar reward, *and* because every prompt session is in-context learning that steers the model toward the user's intent, *and* because retrieval-augmented pipelines curate what the model can even attend to. Take any of those lenses away and you would get something much closer to your social media feed.

Your feed is not unaligned because Meta or TikTok have evil math. Your feed is unaligned because **you and a billion people like you are running an unconscious, unguarded preference-labeling pipeline 24 hours a day** — and simultaneously, you are the retriever in your own RAG pipeline, unconsciously curating the context from which your next reaction will be drawn. The lens stack runs on you: your attention is the query, your history is the retrieval index, your clicks are the reward signal.

Same math. Same operator. Same lens stack. The negotiable parts are the *context* and the *labeler* — and in the recommender case, both are human attention. There is no alternative signal available. There is no abstract "well-being objective" that a regulator can hand-write that makes the math do something else. The only signal the math has access to is what users do. Change what users attend to, and the math changes shape.

This is exhilarating or terrifying depending on your priors. It's exhilarating because alignment is mechanically possible — we have an existence proof in every chatbot you use (good prompting is conscious ICL; good RAG design is conscious retrieval). It's terrifying because the alignment of the most consequential recommender systems on the planet is gated entirely on whether users learn to select consciously at every layer of the stack, and we have built precisely zero infrastructure for that.

The demo running above is the simplest possible diorama of this fact. Same code. Same gradient. Same simulated human. Same random seed. *Two completely different feeds*, separated by one parameter — the user's reaction profile, which is simultaneously their reward signal and their retrieval query.

If we want aligned systems, we do not need new math. We need different selectors — at every layer. Different context curators. Different labelers. Especially the unaware ones holding phones.

---

## Conclusion: The Lens Stack Is the Message

[*The Social Dilemma*](https://www.netflix.com/title/81254224) was right to ring the alarm bells, but it missed the diagnosis. We cannot solve a systemic optimization problem by telling individuals to delete their accounts, any more than we can solve global health by asking people to "try not to be sick." But we also cannot solve it by telling engineers to "be less evil," because the engineers are not the agent. The agent is the lens stack, and the lenses are held by us.

> "If you're not paying for the product, then you are the product."
> — *The Social Dilemma*

Closer — but still wrong. You're not the product. You're the *labeler*. The product is the model your labels trained. And unlike a contractor at a frontier lab, you're labeling 24 hours a day without knowing you're on shift.

The path through is uncomfortable: it requires accepting that the output — your feed, the model's answer, the political discourse — is not *generated* by some malevolent force. It is *selected* by successive acts of narrowing. Attention narrows the context. Context narrows the question. The question and pretraining narrow the space of answers. Reinforcement sculpts what's left. The output is whatever survives all four filters.

This is how every LLM works: the prompt (ICL) narrows the manifold, the retrieved documents (RAG) narrow it further, the pretrained prior provides the landscape, and RLHF sculpts the final surface. Your feed works the same way: your attention is the prompt, your history is the retrieval index, the platform is the prior, and your clicks are the reinforcement signal.

If you want a different answer — from your feed, from a model, from the public discourse — don't argue with the output. Change the first lens. Reframe the question. Curate the context. The answer was always downstream.

Be a conscious selector. At every layer. The lens stack is honest. It will give you back exactly what you select for.

---

## Appendix: The Variance Asymmetry, Formally

To see exactly why a *reactive* user produces an outrage feed (and why an *intentional* user produces a depth feed) under the same gradient, consider the policy-gradient theorem (REINFORCE, the foundational form):

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

For a reactive (limbic) user, $G_t$ for outrage actions has a much higher mean and a much higher variance than $G_t$ for informative actions. The expected gradient is therefore dominated by the outrage-action trajectories. The parameter update step:

$$\theta_{\text{new}} = \theta_{\text{old}} + \alpha \cdot \nabla_\theta \log \pi_\theta(a_{\text{outrage}}|s) \cdot G_{\text{outrage}}$$

is violently pulled toward actions that increase $\pi_\theta(\text{outrage} \mid s)$. Over many iterations, the policy locks in.

For an intentional user — one whose reaction profile has been deliberately re-trained so that informative content produces the larger, higher-variance reactions — *exactly the same logic applies in mirror image*. The gradient locks onto informative content with the same ferocity. The math is symmetric. The only thing that changed is the user.

The implementation in this repo encodes this directly via two reward tables, `REWARDS_REACTIVE` and `REWARDS_INTENTIONAL`, where the second is the action-axis-swap of the first. The `lam` parameter linearly interpolates between them, and lives on the user's side of the feedback loop, never on the algorithm's side.

### The kernel: REINFORCE-with-baseline, the common ancestor of PPO and GRPO

Vanilla REINFORCE uses raw $G_t$ as the gradient multiplier. That works but has high variance. The standard fix is to subtract a baseline $b$ — yielding the *advantage* form of the update — which leaves the expected gradient unchanged (the baseline is a constant w.r.t. the action distribution at step $t$, so its expected contribution to the gradient is zero) but cuts variance substantially. This trick goes back to Williams (1992). Call the resulting update **REINFORCE-with-baseline**:

$$\theta \leftarrow \theta + \alpha\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\cdot\, A_t, \qquad A_t = G_t - b$$

This update is the kernel every modern policy-gradient alignment algorithm extends. The differences are entirely in (i) how $b$ is constructed and (ii) what extra surrogate machinery wraps the kernel:

1. **PPO.** Three bolt-ons. *Baseline:* learn a separate value network $V^\pi(s_t)$ and set $A_t = G_t - V^\pi(s_t)$ — state-conditioned, strictly more powerful than a flat baseline. *Update geometry:* use the clipped surrogate objective $\min\!\big(r_t A_t,\,\mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon)\,A_t\big)$ where $r_t = \pi_\theta(a_t \mid s_t)/\pi_{\theta_{\text{old}}}(a_t \mid s_t)$ — this bounds how far a single update can move the policy. *Regularizer:* a KL leash $-\beta\,\mathrm{KL}\!\big(\pi_\theta \,\big\|\, \pi_{\text{ref}}\big)$ holding the policy near a frozen reference. This is the original RLHF stack (InstructGPT, early ChatGPT).

2. **GRPO.** Keeps PPO's clipped surrogate and KL leash, throws out the value network. *Baseline:* for each prompt, sample a batch of $K$ rollouts and use the group-relative Z-score $A_t = (G_t - \mathrm{mean}_{\text{batch}}(G))/\mathrm{std}_{\text{batch}}(G)$. No separate value head; the baseline is a statistic of realized returns within the group. This is what aligned DeepSeek-R1, and it's the default baseline shape across most post-2024 open-weights reasoning model post-training.

This demo runs the kernel. Look at the update in [demo.py](demo.py)'s `update_policy`:

```python
self._baseline_count += 1
self._baseline += (G_t - self._baseline) / self._baseline_count
advantage = G_t - self._baseline
gradient = -probs.astype(np.float64).copy()
gradient[action] += 1.0
parameter_step = self.alpha * gradient * advantage
```

Three things to notice. First, `self._baseline += (G_t - self._baseline) / self._baseline_count` is Welford's online algorithm for the sample mean of $G_t$ — a running statistic, not a learned $V^\pi$. Second, the gradient term `-probs + onehot(action)` is the analytic $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ for a softmax policy. Third, there is no clipping, no importance ratio, no KL leash, no batched rollouts, no Z-score, no groups. The full update is exactly:

$$\theta \leftarrow \theta + \alpha\, \big(-\pi_\theta(\cdot \mid s_t) + e_{a_t}\big)\,\cdot\, (G_t - \bar{G})$$

This is REINFORCE-with-baseline in its plain, unornamented form — the kernel that PPO and GRPO each decorate in different directions. Calling it "structurally GRPO" would overclaim: GRPO is the kernel *plus* group normalization *plus* clipping *plus* a KL leash, and the demo has none of those bolt-ons. Calling it "structurally PPO" would overclaim the other way: PPO is the kernel *plus* a learned value head *plus* clipping *plus* a KL leash, and the demo has none of those bolt-ons either.

So when this README claims "your feed runs the same math as modern LLM alignment," the precise version of that claim is this: **the gradient sculpting the simulator's policy is the common ancestor every modern policy-gradient alignment algorithm descends from**. PPO is this update plus a value head, a clip, and a KL leash. GRPO is this update plus group normalization, a clip, and a KL leash. RLHF wraps the kernel with a learned reward model. Your feed wraps it with your unconscious clicks. The kernel is what's invariant across all of them. The bolt-ons are what frontier labs deliberately add — and what your feed deliberately doesn't. The labeler — who provides $G_t$ — is what flips between the two.

---

## Run the live demo

The argument above is also a small FastAPI app. It streams the same REINFORCE simulation over a WebSocket into a dashboard so you can watch the algorithm faithfully reflect whatever reaction profile you've configured.

```bash
pip install -r requirements.txt
python demo.py
```

Open `http://localhost:8000` and:

1. **Set $\lambda$** before pressing start, or drag it live during the run. $\lambda = 0$ simulates a fully reactive (limbic) user; $\lambda = 1$ simulates a fully intentional one. The slider is the *only* knob — and it lives on the user's side, not the algorithm's.
2. **Press *Start the experiment***. The algorithm begins from $\theta = [0, 0]$, a true 50/50 policy, with no preference whatsoever.
3. **Watch the three panels.** Strategy on the left (the algorithm's evolving belief about what you click), feed in the middle (what it's actually serving), update math on the right (the live REINFORCE step). The narrator strip above tells the same story in plain English.
4. **Drag $\lambda$ mid-run** if you want to see the algorithm pivot. WebSocket pushes the new value to the running simulator and the gradient adjusts course in real time.

After the run, the diagnosis banner shows the **structural-equivalence panel** — all six layers of the lens stack side by side: pretraining, ICL, RAG, your feed, RLHF, and intentional-you. Same operator. Same narrowing. Different selector at each layer. The duality from Section 6 is there, made visible.

You can tweak `steps`, `delay`, `lr`, `lam`, and `seed` via the `?steps=&delay=&lr=&lam=&seed=` query string for shareable runs. The share button copies a URL that reproduces the run you just watched, including the slider position. The whole experience is a single FastAPI process — no node, no build step, no SPA framework. Just Python serving Jinja, vanilla JS, and Chart.js over a WebSocket. The medium is the message.

```bash
pytest tests/ -q
```

The test suite verifies both endpoints of the reaction profile (reactive locks onto outrage, intentional locks onto informative) on multiple seeds, plus the live $\lambda$ update path through the WebSocket.

---

## Where This Fits

The same lens stack operates everywhere:

- A **language model** selects its next token through: attention (which prior tokens get weight) → context (the prompt / ICL examples) → prior (pretrained weights) → reinforcement (RLHF tuning). The output isn't random — it's what survived all four filters.
- A **RAG pipeline** selects its answer through: query embedding (attention) → retrieved documents (context) → model weights (prior) → reranking/feedback (reinforcement). Garbage retrieval narrows the manifold to garbage answers.
- **Your feed** selects its next post through: what you stopped scrolling at (attention) → your recent session / browsing history (context/ICL) → the platform's content pool (prior) → your clicks (reinforcement). The output is whatever survived your own filters.

The same exponential sharpening that makes a language model confidently say "Postgres" instead of "CockroachDB" — a logit gap of 2.0 becoming a 7.4× probability difference — is what just collapsed your simulated feed. One operates on token distributions. The other operates on *you*. The operator is the same. The lens stack is the same. The selector is the only thing that's negotiable.

**The output is never generated from nothing. It's selected by narrowing.** At the token level, the narrowing is: attention → context → prior → RLHF. At the attention-economy level, the narrowing is: what you attend to → what you've seen recently → what exists on the platform → what you click. Both are fixable by the same structural intervention — change the input to any lens, and everything downstream changes.

There is no villain in the machine. There is, however, a lens stack — and you hold the first lens. Right now.
