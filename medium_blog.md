# Teaching a Neural Network to Sketch — Without Ever Showing It a Pair

### How I built a CycleGAN from scratch to translate between two sketch domains, and what I learned about unpaired image translation along the way.

---

![CycleGAN live Gradio demo — mountain, tree, and tent sketches translated in real time](https://raw.githubusercontent.com/fahadcs321/CycleGan/refs/heads/main/medium_blog.md/10_gradio_app_screenshot.png)

---

Imagine asking a student to learn French by handing them a pile of English books and a pile of French books — separately, with no translations between them. That is essentially what **CycleGAN** does with images.

In this project, I built a CycleGAN from scratch in PyTorch to learn a bidirectional mapping between two sketch domains — detailed art-school sketches from **TU-Berlin** and 20-second mobile doodles from **Google QuickDraw** — without ever showing the model a paired example. The whole thing ran on Kaggle's free dual-T4 GPU tier in under two hours.

By the end of this post, you will understand:

- Why unpaired image-to-image translation is fundamentally different from supervised translation
- How cycle consistency replaces paired supervision
- The practical tricks that separate "GAN technically runs" from "GAN actually converges"
- How the model did on SSIM, PSNR, and visual inspection

If you just want the code, the full notebook and deployed Gradio app are linked at the bottom.

---

## The Problem: Translation Without Parallel Text

Most image translation models are trained on pairs. If you want to colourise a black-and-white photo, you feed the network thousands of (grayscale, colour) pairs and let supervised learning do its thing.

But what if you do not have pairs? What if you have a stack of real-world horse photos and a stack of zebra photos from completely different contexts and cannot possibly match them one-to-one? This is the setting CycleGAN was designed for.

The original CycleGAN paper (Zhu et al., 2017) proposed a wonderfully elegant idea: if you learn **two** generators — one for each direction — you can use the composition of both as a proxy for supervision.

> Translate an image to the other domain, then back. The reconstruction should match the original.

This "cycle consistency" constraint replaces paired data.

---

## The Datasets: Two Flavours of Sketch

I deliberately picked two sketch sources rather than sketch-versus-photo, to make the mapping learnable but not trivial.

**TU-Berlin Sketch Dataset** — 20,000 carefully drawn sketches across 250 object categories (flying saucer, ashtray, scissors, bench…). Each is 1111×1111 pixels, drawn by humans over tens of seconds with detail like internal contours and hatching. I pulled this straight from Hugging Face with `load_dataset("sdiaeyu6n/tu-berlin")`.

**Google QuickDraw** — 50 million stroke-based doodles, each drawn in 20 seconds on a mobile screen by Quick, Draw! players. The dataset ships as JSON stroke arrays which I rasterised to 256×256 PNGs.

I sampled 2,000 TU-Berlin sketches (Domain A) and 1,700 QuickDraw doodles (Domain B).

**TU-Berlin — deliberate, detailed human drawings:**

![TU-Berlin sketch samples — frog, tractor, TV, strawberry, skeleton, foot, car, knife](https://raw.githubusercontent.com/fahadcs321/CycleGan/refs/heads/main/medium_blog.md/04_tu_berlin_samples.png)

**QuickDraw — fast, loose 20-second mobile doodles:**

![QuickDraw doodle samples — simple rounded shapes and quick strokes](https://raw.githubusercontent.com/fahadcs321/CycleGan/refs/heads/main/medium_blog.md/05_quickdraw_samples.png)

The aesthetic difference is immediate: TU-Berlin sketches are deliberate and detailed, QuickDraw doodles are fast and loose.

### Is there actually a domain gap to learn?

This is the question to ask before training any translation model. If your two domains are statistically identical, CycleGAN has nothing meaningful to translate.

I looked at two diagnostics:

![Pixel intensity (log scale) on the left shows both domains are line art. Ink density on the right shows the real difference: TU-Berlin averages 1.8% dark pixels, QuickDraw averages 4.2%](https://raw.githubusercontent.com/fahadcs321/CycleGan/refs/heads/main/medium_blog.md/02_pixel_and_ink_density.png)

On the left, a log-scale pixel intensity histogram. Both domains are pure line art (huge mass at 0 and 255), but the ink tone and distribution differ subtly.

On the right is the more telling signal — **ink density**, the fraction of dark pixels per image. TU-Berlin sits at a mean of 1.8% dark pixels with tight variance. QuickDraw has a mean of 4.2% with a much wider spread. That makes intuitive sense: detailed art-school sketches use controlled, sparse lines; QuickDraw has messier strokes because people are drawing under a 20-second timer.

Those distributions do not overlap much. CycleGAN has room to learn.

---

## What the DataLoader Actually Produces

CycleGAN uses *unpaired* data — meaning the DataLoader returns one random sketch and one independently sampled doodle per step. There is no correspondence between them, and the model never learns one.

![A batch from the unpaired DataLoader — top row is Domain A (TU-Berlin), bottom row is Domain B (QuickDraw). No pair correspondence.](https://raw.githubusercontent.com/fahadcs321/CycleGan/refs/heads/main/medium_blog.md/06_dataloader_batch.png)

This is the crucial mental model: at training time, the network literally does not know which doodle "goes with" which sketch, because nothing does.

---

## The Architecture

CycleGAN uses four networks, trained jointly:

| Network | Role |
|---|---|
| `G_AB` | Generator: Sketch → Doodle |
| `G_BA` | Generator: Doodle → Sketch |
| `D_A` | Discriminator: real sketch vs fake sketch |
| `D_B` | Discriminator: real doodle vs fake doodle |

### Generator — ResNet-6

I followed the original recipe but reduced ResNet blocks from 9 to 6 to fit the Kaggle T4 memory budget:

```
c7s1-64 → d128 → d256 → 6 × ResBlock(256) → u128 → u64 → c7s1-3
```

Every convolution uses **reflection padding** (avoids edge artifacts) and **instance normalization** (per-sample, rather than per-batch — critical for style transfer). The output is passed through `tanh` so it lands in [-1, 1], matching the normalised inputs.

7.84M parameters per generator. Two generators plus two discriminators means roughly 20M trainable parameters total.

### Discriminator — 70×70 PatchGAN

Instead of asking "is this whole image real or fake?", a PatchGAN asks the question over overlapping 70×70 patches and outputs a grid of real/fake scores. At 128×128 input resolution, this gives a 14×14 patch map.

Why? Because **textures are local**. A full-image discriminator often ignores local artifacts to focus on global composition. PatchGAN forces it to scrutinise every region, which sharpens the generator's output.

```python
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        nf, nf_prev = ndf, ndf
        for i in range(1, n_layers):
            nf_prev, nf = nf, min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, 4, 2, 1),
                nn.InstanceNorm2d(nf), nn.LeakyReLU(0.2, True),
            ]
        ...
```

---

## The Losses: Three Signals, One Objective

CycleGAN's total generator loss has three components:

**1. Adversarial loss (LSGAN variant)** — fool the discriminators. I used MSE on patch outputs rather than binary cross-entropy, because LSGAN is visibly more stable and does not saturate.

**2. Cycle-consistency loss** — the core idea. For every image `x` in Domain A, the round-trip `G_BA(G_AB(x))` should match `x`. Mirrored for Domain B. I weighted this with **λ = 10**, the paper's default. This is by far the dominant signal early in training.

**3. Identity loss** — feed a Domain-B image into the `G_AB` generator (which was trained to map A→B). It should come out unchanged, because it is already in the target domain. Weighted **λ = 5**. This stops the generators from inventing spurious colour shifts or tone changes.

Combined: **`L_G = L_GAN + 10 · L_cycle + 5 · L_identity`**

---

## Training: Where GANs Actually Get Tricky

Getting a GAN to converge is a different sport from training a normal supervised model. Three ingredients made the difference:

### 1. Replay buffer

The discriminator gets updated on a mix of real images and a rolling buffer of *past* fake images — not just whatever the generator produced this step. Without this, `D` oscillates badly: it becomes great at detecting whatever `G` output last minute, then `G` shifts its style and `D` has to relearn. With the buffer, `D` sees a stable distribution of fakes and settles.

```python
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.data = []
    def push_and_pop(self, images):
        out = []
        for img in images.detach():
            if len(self.data) < self.max_size:
                self.data.append(img); out.append(img)
            else:
                if random.random() < 0.5:
                    i = random.randint(0, self.max_size - 1)
                    out.append(self.data[i].clone()); self.data[i] = img
                else:
                    out.append(img)
        return torch.cat(out, 0)
```

### 2. Mixed precision on dual GPUs

Kaggle gives you two T4 GPUs for free. Using `torch.cuda.amp` for FP16 forward passes, I wrapped every model in `nn.DataParallel` to split each batch across both cards. This brought epoch time from ~6 minutes to under 3.5.

### 3. Linear LR decay after warmup

Run at the full 2e-4 learning rate for 15 epochs, then linearly decay to zero over the remaining 15. Adam with β = (0.5, 0.999) — the lower momentum β₁ is a standard GAN trick to prevent overshoot.

### Resumable checkpoints

Kaggle sessions time out. I wrote the full state (models + optimisers + schedulers + AMP scalers + loss history) to disk every epoch so training picks up exactly where it left off on the next session. This turned out to be essential — I hit a Kaggle timeout at epoch 28 and resumed seamlessly.

---

## Results

### Training Curves

![Three loss curves across 30 epochs — Generator stabilises around 2.0, Discriminator drops to ~0.1, Cycle-consistency loss falls from 1.24 to 0.28](https://raw.githubusercontent.com/fahadcs321/CycleGan/refs/heads/main/medium_blog.md/07_training_curves.png)

Three plots tell the story:

- **Generator loss** stabilises around 2.0 after the first epoch's wild swings. The generator is not winning outright (loss should not go to zero) but holds its own against the discriminator.
- **Discriminator loss** drops from 0.63 to ~0.1 — `D` is confident, but not perfectly so, which is exactly the equilibrium you want.
- **Cycle-consistency loss** drops from 1.24 to 0.28 — a **4.4× reduction**. This is the actual supervision signal, and it is still trending downward at epoch 30. More epochs would help further.

### Quantitative Evaluation

Because this is *unpaired* translation, there is no ground-truth target for a translated image — you cannot ask "how close is `G_AB(x)` to the true x-in-domain-B" because no such truth exists.

The right metric is the **cycle reconstruction**: given the round-trip `G_BA(G_AB(x))`, how close is the reconstruction to the original? This directly measures what cycle consistency optimises.

| Metric | Domain A (sketch ↺) | Domain B (doodle ↺) |
|---|---|---|
| **SSIM** | 0.9396 | **0.9920** |
| **PSNR** | 27.60 dB | **29.44 dB** |

A cycle SSIM above 0.94 on unpaired data is strong. The reason Domain B scores higher (0.99) is that QuickDraw doodles are sparser and structurally simpler — there is less for the generator to "lose" in the round-trip.

### Qualitative Results

The numbers are fine, but what actually comes out?

**Sketch → Doodle → Sketch**

![Five TU-Berlin sketches (left), their QuickDraw-style translations (middle), and the cycle reconstructions (right)](https://raw.githubusercontent.com/fahadcs321/CycleGan/refs/heads/main/medium_blog.md/08_sketch_to_doodle_translations.png)

Column 1 is the original TU-Berlin sketch, column 2 is the QuickDraw-style translation, column 3 is the reconstruction. Notice how the translated middle column has **thicker, bolder strokes** — that is the QuickDraw style the generator learned. The reconstructions in column 3 recover the original fine detail remarkably well.

**Doodle → Sketch → Doodle**

![Five QuickDraw doodles (left), their TU-Berlin-style translations (middle), and the cycle reconstructions (right)](https://raw.githubusercontent.com/fahadcs321/CycleGan/refs/heads/main/medium_blog.md/09_doodle_to_sketch_translations.png)

This direction is harder — the generator has to *add* detail to a minimal doodle. You can see it attempting to lighten strokes and soften the harsh QuickDraw marks. The reconstructions are near-perfect because the round-trip passes through a detailed intermediate state.

---

## Deployment: A Live Gradio App

A CycleGAN that only runs in a notebook is not very useful. I exported the two generator checkpoints (31 MB each) and wrapped them in a Gradio interface:

```python
def translate(img, direction):
    x = pre(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y = (G_AB if direction == "Sketch → Photo" else G_BA)(x)
    return T.ToPILImage()((y.clamp(-1, 1).squeeze(0).cpu() + 1) / 2)
```

![The deployed Gradio app — upload an image, pick a direction, get the translation inline](https://raw.githubusercontent.com/fahadcs321/CycleGan/refs/heads/main/medium_blog.md/10_gradio_app_screenshot.png)

Upload any sketch, pick a direction, and the translation comes back inline. The whole app is ~80 lines and deploys to a free Hugging Face Space in about 3 minutes.

---

## What I Would Do Differently

**More epochs.** The cycle loss was still dropping at epoch 30. I bet 50 would give visibly sharper translations.

**Wider ResNet.** I stuck to 6 blocks to fit memory, but 9 blocks at 256×256 resolution (the original paper's config) would capture more detail. A single T4 cannot fit it, but you could use gradient accumulation.

**Actual sketch ↔ photo, not sketch ↔ doodle.** Swapping QuickDraw for a real photo dataset (CelebA faces, Oxford Flowers) would make this a genuine sketch-to-photo task — much harder, but also visually more impressive.

**Feature-matching loss.** Adding a VGG-based perceptual loss on top of the L1 cycle loss often sharpens textures at the cost of slightly lower SSIM. Worth experimenting.

---

## Takeaways

1. **Cycle consistency is a remarkable trick.** Replacing paired data with "round-trip should match" expands the set of problems you can solve. Any two domains you can collect data from become candidates.

2. **GAN stabilisation tricks matter more than architecture tweaks.** Reflection padding, replay buffers, LSGAN, and linear LR decay each contribute. Skip any one and training gets noticeably worse.

3. **The "right metric" depends on the supervision.** In paired settings, SSIM against the target makes sense. In unpaired settings, SSIM against the cycle reconstruction is the honest metric — it measures what the loss actually enforces.

4. **Kaggle's free tier is remarkably capable.** Two T4s + AMP + DataParallel got me a full 30-epoch training run in about 90 minutes. For a personal project, that is unreasonably good.

---

## Resources

- **Full Kaggle notebook** (training + evaluation + Gradio): `[link to your Kaggle notebook]`
- **Live Gradio app** on Hugging Face Spaces: `[link to your HF space]`
- **Source code**: `[link to your GitHub]`

Original paper: Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.*

---

*If you found this useful, a clap or a follow would mean a lot. Questions welcome in the responses.*
