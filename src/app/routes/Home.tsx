import {
  FileText,
  Copy,
  Check,
  Database,
  Network,
  TrendingUp,
} from "lucide-react";
import { SiGithub, SiHuggingface } from "react-icons/si";
import { useLocation } from "react-router";
import { useEffect, useState, useCallback } from "react";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../../components/ui/table";
import { Button } from "../../components/ui/button";
import InteractiveExamples from "../../components/InteractiveExamples";
import projectData from "../../data/project.json";

const BASE = import.meta.env.BASE_URL;
import type { Route } from "./+types/Home";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import {
  buildMeta,
  buildScholarlyArticleSchema,
  seoDefaults,
} from "@/lib/seo";

export const meta: Route.MetaFunction = () => {
  const schema = buildScholarlyArticleSchema({
    title: projectData.title,
    description: projectData.seo.description,
    authors: projectData.authors,
    keywords: projectData.seo.keywords,
    url: seoDefaults.SITE_URL,
    image: `${seoDefaults.SITE_URL}${projectData.seo.ogImage}`,
  });

  return [
    ...buildMeta({
      title: projectData.title,
      description: projectData.seo.description,
      path: "/",
      keywords: projectData.seo.keywords,
      image: `${seoDefaults.SITE_URL}${projectData.seo.ogImage}`,
      imageAlt: seoDefaults.DEFAULT_IMAGE_ALT,
      type: "article",
    }),
    {
      tagName: "script",
      type: "application/ld+json",
      children: JSON.stringify(schema),
    },
  ];
};

function CopyBibtexButton() {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(projectData.citation.bibtex).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, []);

  return (
    <Button variant="outline" size="sm" onClick={handleCopy}>
      {copied ? (
        <>
          <Check className="mr-2 h-4 w-4" />
          Copied
        </>
      ) : (
        <>
          <Copy className="mr-2 h-4 w-4" />
          Copy BibTeX
        </>
      )}
    </Button>
  );
}

function Home() {
  const location = useLocation();

  useEffect(() => {
    if (!location.hash) return;
    const element = document.querySelector(location.hash);
    element?.scrollIntoView({ behavior: "smooth" });
  }, [location.hash]);

  return (
    <main className="container mx-auto px-6 py-8 space-y-20 xl:max-w-4xl">
      {/* Hero Section */}
      <section
        className="relative text-center flex flex-col justify-between gap-8 rounded-xl sm:rounded-3xl overflow-hidden w-full"
        style={{
          minHeight: "min(440px, 70vh)",
        }}
      >
        {/* Background montage of point-cloud objects */}
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `url(${BASE}figures/hero-bg.jpg)`,
            backgroundSize: "cover",
            backgroundPosition: "center",
            pointerEvents: "none",
          }}
        />
        {/* Bottom-darkening overlay for text readability */}
        <div
          className="absolute inset-0"
          style={{
            background:
              "linear-gradient(to bottom, transparent 40%, rgba(0, 0, 0, 0.6) 70%, rgba(0, 0, 0, 0.8) 100%)",
            pointerEvents: "none",
          }}
        />

        {/* CVPR 2026 badge (top) */}
        <div className="relative z-10 flex justify-center px-4 pt-6 sm:pt-8">
          <img
            src={`${BASE}figures/cvpr-logo-white.png`}
            alt="CVPR 2026 — Denver, Colorado, June 3–7"
            className="h-9 w-auto opacity-95 sm:h-12"
          />
        </div>

        {/* Content anchored to bottom */}
        <div className="relative z-10 space-y-3 pb-6 sm:pb-8">
            <p className="text-lg text-white sm:text-xl md:text-2xl lg:text-3xl tracking-tight font-semibold px-4">
              {projectData.title}
            </p>

            {/* Authors */}
            <div className="space-y-2">
              <div className="flex flex-wrap justify-center gap-x-3 gap-y-1 text-sm sm:text-base text-white px-2">
                {projectData.authors.map((author, i) => (
                  <span key={i} className="whitespace-nowrap">
                    {author.url ? (
                      <a
                        href={author.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="hover:text-primary"
                      >
                        {author.name}
                      </a>
                    ) : (
                      author.name
                    )}
                    <sup className="text-xs text-gray-300 ml-0.5">
                      {author.affiliations.map((id) => id + 1).join(",")}
                    </sup>
                  </span>
                ))}
              </div>
              <div className="flex flex-wrap justify-center gap-x-3 gap-y-1 text-xs sm:text-sm text-gray-300 px-2">
                {projectData.affiliations.map((aff, i) => (
                  <span key={i}>
                    <sup className="mr-0.5">{i + 1}</sup>
                    {aff.url ? (
                      <a
                        href={aff.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="hover:text-primary"
                      >
                        {aff.name}
                      </a>
                    ) : (
                      aff.name
                    )}
                  </span>
                ))}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-wrap justify-center gap-2 sm:gap-3 px-2">
              <Button variant="outline" disabled>
                <FileText className="mr-2 h-4 w-4" />
                Paper (TBA)
              </Button>
              <Button variant="outline" asChild>
                <a
                  href={projectData.links.code}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <SiGithub className="mr-2 h-4 w-4" />
                  Code
                </a>
              </Button>
              <Button variant="outline" asChild>
                <a
                  href={projectData.links.models}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <SiHuggingface className="mr-2 h-4 w-4" />
                  Models
                </a>
              </Button>
              <Button variant="outline" asChild>
                <a
                  href={projectData.links.dataset}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Database className="mr-2 h-4 w-4" />
                  Dataset
                </a>
              </Button>
            </div>
          </div>
      </section>

      {/* Teaser */}
      <section className="-mt-8 space-y-3">
        <img
          src={`${BASE}figures/teaser.png`}
          alt="Existing 3D-LLMs and 2D VLMs struggle with multi-object 3D comparison, while our Multi-3DLLM compares geometry across objects."
          className="mx-auto w-full rounded-xl border bg-white shadow-sm"
        />
        <p className="text-center text-sm text-muted-foreground">
          Existing 3D-LLMs and 2D VLMs fail at detailed multi-object comparison.
          Multi-3DLLM, trained on our MO3D dataset (~70k QA pairs), reasons about
          geometry <em>across</em> objects.
        </p>
      </section>

      {/* Key Message */}
      <section className="rounded-xl border bg-card p-8">
        <p className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-3">
          Key Message
        </p>
        <blockquote className="space-y-1">
          {projectData.abstract.message.map((line, i) => (
            <p
              key={i}
              className={
                i === projectData.abstract.message.length - 1
                  ? "text-lg font-medium leading-relaxed text-foreground"
                  : "text-lg italic leading-relaxed text-muted-foreground"
              }
            >
              {line}
            </p>
          ))}
        </blockquote>
      </section>

      {/* Abstract */}
      <section id="abstract" className="space-y-4">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
          Abstract
        </h2>
        <p className="leading-relaxed text-muted-foreground">
          {projectData.abstract.problem}{" "}
          {projectData.abstract.solution}{" "}
          {projectData.abstract.results}
        </p>
        <p className="border-l-2 border-primary/50 pl-4 leading-relaxed text-foreground">
          {projectData.abstract.scope}
        </p>
      </section>

      {/* Key Contributions */}
      <section id="contributions" className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
          Key Contributions
        </h2>
        <div className="grid gap-6 md:grid-cols-3">
          {projectData.contributions.map((contribution, i) => {
            const icons = [Database, Network, TrendingUp];
            const Icon = icons[i];
            return (
              <div
                key={i}
                className="rounded-xl border bg-card p-6 space-y-3"
              >
                <div className="flex items-center gap-3">
                  <div className="rounded-lg bg-primary/10 p-2.5">
                    <Icon className="h-5 w-5 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold">
                    {contribution.title}
                  </h3>
                </div>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {contribution.description}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      {/* MO3D Dataset & Mini-Apps Benchmarks */}
      <section id="dataset" className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
          MO3D Dataset & Mini-Apps Benchmarks
        </h2>
        <p className="leading-relaxed text-muted-foreground">
          MO3D (Multi-Object in 3D) is an instruction-tuning dataset with ~70k
          high-quality QA pairs, designed for multi-object comparison across
          three task types: Positional, Comparative, and Holistic understanding.
          Alongside MO3D, we introduce two application-driven benchmarks: Shape
          Mating (SM) for geometric compatibility and Change Captioning (CC) for
          edit-grounded understanding.
        </p>

        {/* Interactive examples gallery */}
        <div className="space-y-2">
          <h3 className="text-lg font-semibold">Explore Examples</h3>
          <p className="text-sm text-muted-foreground">
            Real QA pairs across the three task families. Switch tabs and
            examples, or rotate the colored point clouds in 3D.
          </p>
          <InteractiveExamples />
        </div>

        <div className="space-y-4">
          <h3 className="text-lg font-semibold">MO3D Task Types</h3>
          <div className="grid gap-4 sm:grid-cols-3">
            {projectData.benchmarks.mo3d.tasks.map((task, i) => (
              <div key={i} className="rounded-lg border p-4 space-y-1">
                <p className="font-medium">{task.name}</p>
                <p className="text-sm text-muted-foreground">
                  {task.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="rounded-lg border p-4 space-y-1">
            <p className="font-medium">Shape Mating (SM)</p>
            <p className="text-sm text-muted-foreground">
              {projectData.benchmarks.shapeMating.description}
            </p>
          </div>
          <div className="rounded-lg border p-4 space-y-1">
            <p className="font-medium">Change Captioning (CC)</p>
            <p className="text-sm text-muted-foreground">
              {projectData.benchmarks.changeCaptioning.description}
            </p>
          </div>
        </div>
      </section>

      {/* Model Architecture */}
      <section id="architecture" className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
          Multi-3DLLM Architecture
        </h2>
        <p className="text-muted-foreground leading-relaxed">
          Multi-3DLLM extends PointLLM with a lightweight Patch-Interaction
          Transformer (PIT) that enables cross-object geometric reasoning while
          preserving fine-grained local geometry. The architecture processes
          multiple point clouds independently, then applies patch-level
          self-attention to capture inter-object and intra-object relationships
          before passing enhanced tokens to the LLM.
        </p>
        {/* Architecture diagram */}
        <figure className="mx-auto max-w-3xl space-y-2">
          <img
            src={`${BASE}figures/architecture.png`}
            alt="Multi-3DLLM architecture: each point cloud is encoded and projected independently, a Patch-Interaction Transformer applies cross- and intra-object attention, and enhanced patch tokens are passed to the LLM."
            className="w-full rounded-xl border bg-white shadow-sm"
          />
          <figcaption className="text-center text-sm text-muted-foreground">
            Each point cloud is encoded independently; the Patch-Interaction
            Transformer (MHSA + FFN with a learned scaler gate) injects
            cross-object interaction before the enhanced tokens reach the frozen
            LLM.
          </figcaption>
        </figure>
      </section>

      {/* Benchmark Results */}
      <section id="results" className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
          Benchmark Results
        </h2>

        {/* MO3D Results */}
        <div className="space-y-4">
          <h3 className="text-xl font-semibold">
            MO3D Multi-Object Comparison
          </h3>
          <ScrollArea className="w-full rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="min-w-[180px]">Model</TableHead>
                  <TableHead className="text-center">Positional (%)</TableHead>
                  <TableHead className="text-center">
                    Comparative (%)
                  </TableHead>
                  <TableHead className="text-center">Holistic (%)</TableHead>
                  <TableHead className="text-center font-semibold border-l">
                    Overall (%)
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <TableRow className="bg-primary/10 font-semibold">
                  <TableCell>Multi-3DLLM (Ours)</TableCell>
                  <TableCell className="text-center">
                    {projectData.results.mo3d.ours.positional}
                  </TableCell>
                  <TableCell className="text-center">
                    {projectData.results.mo3d.ours.comparative}
                  </TableCell>
                  <TableCell className="text-center">
                    {projectData.results.mo3d.ours.holistic}
                  </TableCell>
                  <TableCell className="text-center font-bold border-l">
                    {projectData.results.mo3d.ours.overall}
                  </TableCell>
                </TableRow>
                {Object.entries(projectData.results.mo3d.baselines).map(
                  ([model, scores]) => (
                    <TableRow key={model}>
                      <TableCell className="font-medium">{model}</TableCell>
                      <TableCell className="text-center">
                        {scores.positional}
                      </TableCell>
                      <TableCell className="text-center">
                        {scores.comparative}
                      </TableCell>
                      <TableCell className="text-center">
                        {scores.holistic}
                      </TableCell>
                      <TableCell className="text-center border-l">
                        {scores.overall}
                      </TableCell>
                    </TableRow>
                  ),
                )}
              </TableBody>
            </Table>
            <ScrollBar orientation="horizontal" />
          </ScrollArea>
        </div>

        {/* Mini-Apps & Zero-Shot */}
        <div className="grid gap-6 md:grid-cols-3">
          <div className="rounded-xl border bg-card p-6 space-y-3">
            <h3 className="font-semibold">Shape Mating</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Selection</span>
                <span className="font-medium">
                  {projectData.results.shapeMating.ours.selection}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Reasoning</span>
                <span className="font-medium">
                  {projectData.results.shapeMating.ours.reasoning}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Verification</span>
                <span className="font-medium">
                  {projectData.results.shapeMating.ours.verify}%
                </span>
              </div>
            </div>
          </div>

          <div className="rounded-xl border bg-card p-6 space-y-3">
            <h3 className="font-semibold">Change Captioning</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Verification</span>
                <span className="font-medium">
                  {projectData.results.changeCaptioning.ours.verify}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Delta Caption</span>
                <span className="font-medium">
                  {projectData.results.changeCaptioning.ours.deltaCaption}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Overall</span>
                <span className="font-medium">
                  {projectData.results.changeCaptioning.ours.overall}%
                </span>
              </div>
            </div>
          </div>

          <div className="rounded-xl border bg-card p-6 space-y-3">
            <h3 className="font-semibold">Zero-Shot Classification</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">
                  Multi-3DLLM (Ours)
                </span>
                <span className="font-bold text-primary">
                  {projectData.results.zeroShot.ours}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Ours w/o PIT</span>
                <span className="font-medium">
                  {projectData.results.zeroShot.oursWithoutPIT}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">PointLLM</span>
                <span className="font-medium">
                  {projectData.results.zeroShot.pointllm}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Citation */}
      <section id="citation" className="space-y-6">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
          Citation
        </h2>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              If you find our work useful, please cite:
            </p>
            <CopyBibtexButton />
          </div>
          <pre className="overflow-x-auto rounded-xl border bg-card p-4 text-sm">
            <code>{projectData.citation.bibtex}</code>
          </pre>
        </div>
      </section>
    </main>
  );
}

export default Home;
