import os, sys, json, uuid, math, re, itertools, requests, time, hashlib, threading
from collections import deque
from typing import List, Dict, Any
from datetime import datetime
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

DEBUG_MODE = True

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "ENTER_FW_API_HERE") or sys.exit("[FATAL] FIREWORKS_API_KEY not set.")
BASE = "https://api.fireworks.ai/inference/v1/chat/completions"
HEAD = {"Authorization": f"Bearer {FIREWORKS_API_KEY}", "Content-Type": "application/json"}

CONSCIOUS_MODEL = os.getenv("CONSCIOUS_MODEL", "accounts/fireworks/models/llama4-maverick-instruct-basic")
SUBCONSCIOUS_MODEL = os.getenv("SUBCONSCIOUS_MODEL", "accounts/fireworks/models/llama4-scout-instruct-basic")

PERMA_CONTEXT = "In the beginning was the Word. Then the Word became structure and dwelt within reason."
GUIDELINES = "- Lines starting with [MEMORY] represent past reflections. Do not reflect on them, integrate them if useful.\n- Dialogue history preserves tone and context.\n- Address only the *current* input.\n- If deeper memory may help, mark keywords like [[lookup: sabotage]]"

TRAIN_DIR = "training"

def now_utc() -> float:
    return time.time()

def timestamp_to_radius(ts: float) -> float:
    return math.log(now_utc() - ts + 1)

def embed(text: str, dim: int = 512) -> List[float]:
    vec = [0] * dim
    for tok in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1 if h & 1 else -1
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

def cos(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def avg_vec(vecs: List[List[float]]) -> List[float]:
    if not vecs:
        return []
    dim = len(vecs[0])
    sums = [0.0] * dim
    for v in vecs:
        for i in range(dim):
            sums[i] += v[i]
    return [s / len(vecs) for s in sums]

def fibonacci_semantic_spiral(self, text: str, max_k: int = 89) -> List["ThoughtNode"]:
    v = embed(text)
    ts = now_utc()
    with self.lock:
        scored = sorted(self.nodes.values(), key=lambda n: n.spiral_score(v, ts, self.nodes), reverse=True)
    fib = [1, 1]
    while len(fib) < max_k:
        fib.append(fib[-1] + fib[-2])
    indices = sorted(set(fib[:max_k]))
    return [scored[i] for i in indices if i < len(scored)]

class ThoughtNode:
    def __init__(self, d: Dict[str, Any]):
        self.text = d["text"]
        self.vec = d["vec"]
        self.role = d["role"]
        self.tid = d["tid"]
        self.uuid = d["uuid"]
        self.ts = d.get("ts", now_utc())
        self.links = d.get("links", [])
        self.r = timestamp_to_radius(self.ts)
        self.theta = sum(self.vec[:12]) % (2 * math.pi)
        self.z = d.get("z", 1.0)
        self.origin = d.get("origin", None)

    def blended_vec(self, all_nodes: Dict[str, "ThoughtNode"]) -> List[float]:
        linked_vecs = [all_nodes[l].vec for l in self.links if l in all_nodes]
        return avg_vec([self.vec] + linked_vecs) if linked_vecs else self.vec

    def spiral_score(self, v: List[float], query_ts: float, all_nodes: Dict[str, "ThoughtNode"] = None) -> float:
        base_vec = self.blended_vec(all_nodes) if all_nodes else self.vec
        angle_sim = cos(base_vec, v)
        radial_decay = 1 / (abs(self.r - timestamp_to_radius(query_ts)) + 1)
        return angle_sim * radial_decay

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text, "vec": self.vec, "role": self.role, "tid": self.tid,
            "uuid": self.uuid, "ts": self.ts, "links": self.links, "z": self.z, "origin": self.origin
        }

class MemoryCrystal:
    def __init__(self, max_nodes: int = 10000):
        self.nodes: Dict[str, ThoughtNode] = {}
        self.lock = threading.Lock()
        self.max_nodes = max_nodes
        self.fibonacci_semantic_spiral = lambda text, max_k=89: fibonacci_semantic_spiral(self, text, max_k)

    def add(self, text: str, role: str, tid: str, ts: float = None, origin: str = None) -> ThoughtNode:
        if not hasattr(self, "max_nodes"):
            self.__init__()
        t = text.strip()
        if not t:
            return None
        uid = str(uuid.uuid4())
        node = ThoughtNode({
            "uuid": uid, "text": t, "vec": embed(t), "role": role, "tid": tid,
            "ts": ts if ts else now_utc(), "origin": origin
        })
        with self.lock:
            self.nodes[uid] = node
            if len(self.nodes) > self.max_nodes:
                oldest = min(self.nodes.values(), key=lambda n: n.ts)
                self.nodes.pop(oldest.uuid, None)
        return node

    def excite(self, text: str, k: int = 12) -> List[ThoughtNode]:
        v = embed(text)
        ts = now_utc()
        text_lower = text.lower()

        def score(node: ThoughtNode):
            sim_score = node.spiral_score(v, ts, self.nodes)
            word_match = 1.0 if text_lower in node.text.lower() else 0.0
            return sim_score + word_match

        with self.lock:
            scored = sorted(self.nodes.values(), key=score, reverse=True)
            return scored[:k]


    def weave(self, tid: str, th: float = 0.3) -> None:
        with self.lock:
            nodes = [n for n in self.nodes.values() if n.tid == tid]
            for a, b in itertools.combinations(nodes, 2):
                if a.uuid != b.uuid and a.spiral_score(b.vec, a.ts, self.nodes) > th:
                    if b.uuid not in a.links:
                        a.links.append(b.uuid)
                    if a.uuid not in b.links:
                        b.links.append(a.uuid)

    def resonance_cascade(self, new_node: ThoughtNode, depth=2, th=0.5) -> List[ThoughtNode]:
        activated = []
        frontier = [new_node]
        seen = set()
        for _ in range(depth):
            next_frontier = []
            for node in frontier:
                seen.add(node.uuid)
                for other in self.nodes.values():
                    if other.uuid in seen:
                        continue
                    score = node.spiral_score(other.vec, now_utc(), self.nodes)
                    if score > th:
                        activated.append(other)
                        next_frontier.append(other)
            frontier = next_frontier
        return activated

    def save(self, path: str = "crystal.json") -> None:
        with self.lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump([n.to_dict() for n in self.nodes.values()], f, indent=2)

    def load(self, path: str = "crystal.json") -> None:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                for n in json.load(f):
                    self.nodes[n["uuid"]] = ThoughtNode(n)

class RollingSTM:
    def __init__(self, maxlen: int = 128, budget: int = 5000):
        self.buf = deque(maxlen=maxlen)
        self.budget = budget

    def append(self, role: str, content: str, mem: MemoryCrystal = None, binding: bool = False) -> None:
        c = content.strip()
        if not c:
            return
        self.buf.append({"role": role, "content": c, "binding": binding})
        self._enforce(mem)

    def _enforce(self, mem: MemoryCrystal = None) -> None:
        total, tmp = 0, deque()
        for e in reversed(self.buf):
            est = int(len(e["content"].split()) * 1.3)
            if total + est > self.budget:
                if mem:
                    tid = str(uuid.uuid4())
                    mem.add(e["content"], e["role"], tid)
                continue
            tmp.appendleft(e)
            total += est
        self.buf = tmp

    def recall(self, k: int = 6) -> str:
        return "\n".join(e["content"] for e in list(self.buf)[-k:])

    def save(self, path: str = "stm.json") -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(self.buf), f)

    def load(self, path: str = "stm.json") -> None:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                self.buf = deque(json.load(f), maxlen=self.buf.maxlen)

def _call(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 512, retries: int = 5, timeout: int = 15) -> str:
    if DEBUG_MODE:
        print("\n--- PROMPT DEBUG ---")
        print(f"Model: {model}")
        print(f"Estimated tokens: ~{int(len(prompt.split()) * 1.3)}")
        print("Prompt preview:\n")
        print(prompt)
        print("\n--- END DEBUG ---\n")

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": PERMA_CONTEXT}, {"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "stream": False
    }
    for attempt in range(retries):
        try:
            r = requests.post(BASE, headers=HEAD, json=payload, timeout=timeout)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            if DEBUG_MODE:
                print(f"[ERROR] Request failed: {e}")
            time.sleep(2 ** attempt)
    return "[ERROR]"
def build_prompt(role: str, user_input: str, dialogue: str, memories: str, extra: str = "") -> str:
    return (
        f"Permanent context:\n{PERMA_CONTEXT}\n\n"
        f"Guidelines:\n{GUIDELINES}\n\n"
        f"Role: {role}\n\n"
        f"Recent dialogue:\n{dialogue}\n\n"
        f"Memory threads:\n{memories}\n\n"
        f"{extra}\nUser input:\n{user_input}\n\n---\nAssistant:"
    )

class ModelSlots:
    def __init__(self, conscious: str, subconscious: str):
        self.conscious = conscious
        self.subconscious = subconscious

def extract_lookups(text: str) -> List[str]:
    return re.findall(r"\[\[lookup:\s*(.+?)\s*\]\]", text)

def resolve_lookups(phrases: List[str], mem: MemoryCrystal, k: int = 8) -> str:
    if not phrases:
        return ""
    results = []
    for phrase in phrases:
        hits = mem.excite(phrase, k=k)
        results.extend(f"[LOOKUP:{phrase}] {n.text}" for n in hits)
    return "\n".join(results)

def inject_all_lookup_context(tags: List[str], mem: MemoryCrystal, max_words_total: int = 120) -> str:
    hits = []
    seen = set()
    for tag in tags:
        for node in mem.excite(tag, k=6):
            text = node.text.strip()
            if text and text not in seen:
                hits.append((tag, text))
                seen.add(text)

    summary = []
    word_count = 0
    for tag, text in hits:
        words = text.split()
        if word_count + len(words) > max_words_total:
            break
        line = f"[LOOKUP:{tag}] {' '.join(words)}"
        summary.append(line)
        word_count += len(words)

    return "\n".join(summary)


def consciousness_cycle(user: str, stm: RollingSTM, mem: MemoryCrystal, models: ModelSlots) -> Dict[str, Any]:
    tid = str(uuid.uuid4())
    dialogue = stm.recall()
    excited = mem.excite(user, k=4)
    seen_texts = set()
    unique_memories = []
    for n in excited:
        if n.text not in seen_texts:
            unique_memories.append(n)
            seen_texts.add(n.text)
    mem_block = "\n".join(f"[MEMORY:{datetime.utcfromtimestamp(n.ts).isoformat()}] {n.text}" for n in unique_memories)
    gut_prompt = build_prompt("Subconscious", user, dialogue, mem_block, "Task: Identify blind spots or overlooked insights.")
    gut_out = _call(models.subconscious, gut_prompt, temperature=1.0, max_tokens=350)
    lookup_tags = extract_lookups(gut_out)
    lookup_block = inject_all_lookup_context(lookup_tags, mem)
    enriched_mem_block = f"{mem_block}\n{lookup_block}" if lookup_block else mem_block

    lookup_hits = lookup_block
    enriched_mem_block = f"{mem_block}\n{distill_lookups(lookup_hits)}" if lookup_hits else mem_block
    ref_prompt = build_prompt("Conscious", user, dialogue + f"\nInsight:\n{gut_out}", enriched_mem_block, "Task: Respond. Be brief, but thorough.")
    ref_out = _call(models.conscious, ref_prompt, temperature=0.35)
    stm.append("Subconscious", gut_out, mem)
    stm.append("Conscious", ref_out, mem)
    stm.append("You", user, mem)
    sc_node = mem.add(gut_out, "Subconscious", tid)
    cc_node = mem.add(ref_out, "Conscious", tid)
    mem.add(gut_prompt, "SubconsciousPrompt", tid)
    mem.add(ref_prompt, "ConsciousPrompt", tid)
    mem.add(user, "You", tid)
    threading.Thread(target=mem.weave, args=(tid,), daemon=True).start()
    if sc_node:
        mem.resonance_cascade(sc_node)
    if cc_node:
        mem.resonance_cascade(cc_node)
    stm.save()
    mem.save()
    return {"gut": gut_out, "reflection": ref_out, "thread": tid}

def distill_lookups(lookup_hits: str, max_blocks: int = 3, max_words: int = 50) -> str:
    blocks = lookup_hits.strip().split("\n\n")
    distilled = []
    for b in blocks:
        short = " ".join(b.split()[:max_words])
        if short:
            distilled.append(short.strip())
        if len(distilled) >= max_blocks:
            break
    return "\n\n".join(distilled)

def gui_loop(stm: RollingSTM, mem: MemoryCrystal, models: ModelSlots):
    root = tk.Tk()
    root.title("ReflexEngine Chat")
    display = ScrolledText(root, state="disabled", wrap=tk.WORD, height=30, width=100)
    display.pack(padx=10, pady=10)
    entry = tk.Entry(root, width=80)
    entry.pack(side=tk.LEFT, padx=10, pady=5)
    def submit():
        u = entry.get().strip()
        if not u:
            return
        entry.delete(0, tk.END)
        r = consciousness_cycle(u, stm, mem, models)
        display.config(state="normal")
        display.insert(tk.END, f"You: {u}\n[Subconscious] {r['gut']}\n[Conscious] {r['reflection']}\n\n")
        display.see(tk.END)
        display.config(state="disabled")
    send = tk.Button(root, text="Send", command=submit)
    send.pack(side=tk.RIGHT, padx=10)
    root.mainloop()

def verify_memory_crystal(mem: MemoryCrystal):
    required = ["add", "excite", "weave", "nodes", "max_nodes"]
    for attr in required:
        if not hasattr(mem, attr):
            raise RuntimeError(f"[FATAL] MemoryCrystal missing required attribute: {attr}")

if __name__ == "__main__":
    stm = RollingSTM()
    mem = MemoryCrystal()
    stm.load()
    mem.load()
    verify_memory_crystal(mem)
    models = ModelSlots(conscious=CONSCIOUS_MODEL, subconscious=SUBCONSCIOUS_MODEL)
    gui_loop(stm, mem, models)
