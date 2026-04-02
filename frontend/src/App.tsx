import { createResource, createSignal, For, Show } from 'solid-js';

type FrontendConfig = {
  defaultConfigPath: string;
  backend: string;
  demoModels: string[];
  defaultPrompt: string;
};

type RunResponse = {
  run_id: string;
  status: string;
};

type RunListItem = {
  run_id: string;
  status: string;
  paths: Record<string, string>;
};

type RunDetail = {
  run: {
    run_id: string;
    status: string;
    paths: Record<string, string>;
  };
  training: {
    mode: string;
    summary: string;
    metrics: Record<string, string | number>;
    artifacts: Record<string, string>;
    warnings?: string[];
  };
  prompts: {
    meta_prompt: string;
    generation_prompt: string;
    grading_prompt: string;
    prompt_source: string;
  };
  report: {
    backend: string;
    generated_examples: number;
    passed_examples: number;
    training_status: string;
    training_mode: string;
    summary: string;
    warnings: string[];
    demo: {
      meta_prompt: string;
      input_prompt: string;
      grading_prompt: string;
      prompt_source: string;
      before_auto_tuning: string;
      after_auto_tuning: string;
      recommended_small_models: string[];
      notes: string[];
    };
  };
};

async function fetchFrontendConfig(): Promise<FrontendConfig> {
  const response = await fetch('/api/frontend-config');
  return response.json();
}

async function fetchRuns(): Promise<RunListItem[]> {
  const response = await fetch('/api/runs');
  return response.json();
}

export default function App() {
  const [config] = createResource<FrontendConfig>(fetchFrontendConfig);
  const [runs, { refetch }] = createResource<RunListItem[]>(fetchRuns);
  const [configPath, setConfigPath] = createSignal('examples/sample_experiment.yaml');
  const [runId, setRunId] = createSignal('');

  const [run] = createResource(runId, async (id: string): Promise<RunDetail | null> => {
    if (!id) return null;
    const response = await fetch(`/api/runs/${id}`);
    return response.json() as Promise<RunDetail>;
  });

  const submitRun = async () => {
    const response = await fetch(`/api/runs?config_path=${encodeURIComponent(configPath())}`, {
      method: 'POST',
    });
    const payload = (await response.json()) as RunResponse;
    setRunId(payload.run_id);
    await refetch();
  };

  const deleteRun = async () => {
    if (!runId()) return;
    await fetch(`/api/runs/${runId()}`, { method: 'DELETE' });
    setRunId('');
    await refetch();
  };

  const exportRun = () => {
    if (!runId()) return;
    window.open(`/api/runs/${runId()}/export`, '_blank');
  };

  const downloadReport = () => {
    if (!runId()) return;
    window.open(`/api/runs/${runId()}/download/report.json`, '_blank');
  };

  return (
    <main class="min-h-screen bg-slate-950 text-slate-100">
      <div class="mx-auto flex max-w-6xl flex-col gap-6 px-6 py-10">
        <header class="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl">
          <p class="text-sm uppercase tracking-[0.3em] text-cyan-400">auto-tuner</p>
          <h1 class="mt-2 text-3xl font-semibold">Synthetic tuning pipeline</h1>
          <p class="mt-3 max-w-3xl text-sm text-slate-300">
            User meta-prompt → OpenRouter-generated generation/grading prompts → dataset → training run.
          </p>
        </header>

        <section class="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <div class="space-y-6">
            <div class="rounded-2xl border border-slate-800 bg-slate-900 p-6">
              <h2 class="text-lg font-medium">Run pipeline</h2>
              <Show when={config()}>
                {(loaded: () => FrontendConfig) => (
                  <div class="mt-3 space-y-2 text-sm text-slate-400">
                    <p>
                      Default backend: <span class="text-slate-100">{loaded().backend}</span>
                    </p>
                    <p>
                      Initial meta-prompt:
                      <span class="mt-1 block rounded-lg bg-slate-950 p-3 font-mono text-xs text-slate-200">
                        {loaded().defaultPrompt}
                      </span>
                    </p>
                  </div>
                )}
              </Show>
              <label class="mt-5 block text-sm text-slate-300">
                Config path
                <input
                  value={configPath()}
                  onInput={(event: InputEvent & { currentTarget: HTMLInputElement }) => setConfigPath(event.currentTarget.value)}
                  class="mt-2 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm outline-none transition focus:border-cyan-500"
                />
              </label>
              <button
                onClick={submitRun}
                class="mt-4 inline-flex rounded-xl bg-cyan-500 px-4 py-3 text-sm font-medium text-slate-950 transition hover:bg-cyan-400"
              >
                Start run
              </button>
            </div>

            <div class="rounded-2xl border border-slate-800 bg-slate-900 p-6">
              <h2 class="text-lg font-medium">Runs</h2>
              <Show when={runs()} fallback={<p class="mt-3 text-sm text-slate-400">Loading runs…</p>}>
                {(items: () => RunListItem[]) => (
                  <div class="mt-4 space-y-3">
                    <For each={items()}>
                      {(item: RunListItem) => (
                        <button
                          onClick={() => setRunId(item.run_id)}
                          class="flex w-full items-center justify-between rounded-xl border border-slate-800 bg-slate-950 px-4 py-3 text-left text-sm hover:border-cyan-500"
                        >
                          <span class="font-mono text-cyan-300">{item.run_id}</span>
                          <span class="text-slate-400">{item.status}</span>
                        </button>
                      )}
                    </For>
                  </div>
                )}
              </Show>
            </div>
          </div>

          <div class="rounded-2xl border border-slate-800 bg-slate-900 p-6">
            <h2 class="text-lg font-medium">Current run</h2>
            <Show when={runId()} fallback={<p class="mt-3 text-sm text-slate-400">No run started yet.</p>}>
              <div class="mt-4 space-y-4 text-sm">
                <div class="flex flex-wrap gap-3">
                  <button onClick={downloadReport} class="rounded-xl border border-slate-700 px-3 py-2 hover:border-cyan-500">Download report</button>
                  <button onClick={exportRun} class="rounded-xl border border-slate-700 px-3 py-2 hover:border-cyan-500">Export run</button>
                  <button onClick={deleteRun} class="rounded-xl border border-rose-700 px-3 py-2 text-rose-300 hover:bg-rose-950">Delete run</button>
                </div>
                <p>
                  Run ID: <span class="text-cyan-300">{runId()}</span>
                </p>
                <Show when={run() ?? undefined} fallback={<p class="text-slate-400">Loading run details…</p>}>
                  {(value: () => RunDetail) => {
                    const detail = value();
                    return (
                      <>
                        <div class="space-y-1">
                          <p>Status: {detail.run.status}</p>
                          <p>Backend: {detail.report.backend}</p>
                          <p>Mode: {detail.report.training_mode}</p>
                          <p>Prompt source: {detail.prompts.prompt_source}</p>
                          <p>Generated examples: {detail.report.generated_examples}</p>
                          <p>Passed examples: {detail.report.passed_examples}</p>
                        </div>

                        <div>
                          <h3 class="font-medium text-slate-200">Prompts</h3>
                          <div class="mt-3 space-y-3">
                            <pre class="overflow-x-auto rounded-xl bg-slate-950 p-3 text-xs text-slate-200">{detail.prompts.meta_prompt}</pre>
                            <pre class="overflow-x-auto rounded-xl bg-slate-950 p-3 text-xs text-cyan-200">{detail.prompts.generation_prompt}</pre>
                            <pre class="overflow-x-auto rounded-xl bg-slate-950 p-3 text-xs text-amber-200">{detail.prompts.grading_prompt}</pre>
                          </div>
                        </div>

                        <div>
                          <h3 class="font-medium text-slate-200">Training summary</h3>
                          <p class="mt-2 text-slate-300">{detail.training.summary}</p>
                        </div>

                        <div>
                          <h3 class="font-medium text-slate-200">Before / after</h3>
                          <div class="mt-3 grid gap-3">
                            <div>
                              <p class="mb-1 text-xs uppercase tracking-wide text-slate-400">Before</p>
                              <pre class="overflow-x-auto rounded-xl bg-slate-950 p-3 text-xs text-rose-200">{detail.report.demo.before_auto_tuning}</pre>
                            </div>
                            <div>
                              <p class="mb-1 text-xs uppercase tracking-wide text-slate-400">After</p>
                              <pre class="overflow-x-auto rounded-xl bg-slate-950 p-3 text-xs text-emerald-200">{detail.report.demo.after_auto_tuning}</pre>
                            </div>
                          </div>
                        </div>

                        <div>
                          <h3 class="font-medium text-slate-200">Small model examples</h3>
                          <ul class="mt-2 list-disc space-y-1 pl-5 text-slate-300">
                            <For each={detail.report.demo.recommended_small_models}>
                              {(model: string) => <li>{model}</li>}
                            </For>
                          </ul>
                        </div>
                      </>
                    );
                  }}
                </Show>
              </div>
            </Show>
          </div>
        </section>
      </div>
    </main>
  );
}
