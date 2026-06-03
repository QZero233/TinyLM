import heapq
import json
import os
from typing import Optional, List, BinaryIO, Dict, Tuple, Iterable, Iterator
import regex as re
import pickle


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

class BPETokenizer:
    def __init__(self, vocab: Optional[dict[int, bytes]],
                 merges: Optional[list[tuple[bytes, bytes]]],
                 special_token: Optional[List[str]]):
        self.vocab = vocab if vocab is not None else {}
        self.merges = merges if merges is not None else []
        self.special_tokens: List[str] = []
        self.pre_token_encode_result_cache: Dict[Tuple[bytes, ...], List[int]] = {}
        self._append_user_special_tokens(special_token)

    def _append_user_special_tokens(self, special_tokens: Optional[List[str]], check_existence: bool = False):
        if special_tokens is None:
            return

        reverse_vocab: Dict[bytes, int] = {self.vocab[key]: key for key in self.vocab}
        for special_token in special_tokens:
            special_token_bytes = special_token.encode()
            assert not check_existence or special_token_bytes in reverse_vocab
            if special_token_bytes not in reverse_vocab:
                self.special_tokens.append(special_token)
                self.vocab[len(self.vocab)] = special_token_bytes
            if special_token not in self.special_tokens:
                self.special_tokens.append(special_token)


    @classmethod
    def train_bpe(cls, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]):
        self = cls(None, None, special_tokens)
        assert len(self.vocab) == 0 and len(self.merges) == 0, "Must train an empty tokenizer"

        # 添加special tokens
        for special_token in special_tokens:
            self.vocab[len(self.vocab)] = special_token.encode("utf-8")
        # 添加固定的256个bytes
        for i in range(256):
            self.vocab[len(self.vocab)] = bytes([i])

        # 开始train
        # Pre-tokenization
        pre_tokenization_map: Dict[Tuple[bytes, ...], int] = {}
        pre_tokenization_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        num_processes = 4
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk_str = f.read(end - start).decode("utf-8", errors="ignore")
                chunk_split = re.split("|".join([re.escape(token) for token in special_tokens]), chunk_str)
                for chunk_part in chunk_split:
                    pre_tokens: List[str] = re.findall(pre_tokenization_pat, chunk_part)
                    for token in pre_tokens:
                        token_bytes = tuple((bytes([b]) for b in token.encode()))
                        assert len(token_bytes) > 0
                        if token_bytes in pre_tokenization_map:
                            pre_tokenization_map[token_bytes] += 1
                        else:
                            pre_tokenization_map[token_bytes] = 1

        # 开始Merge
        while len(self.vocab) < vocab_size:
            # 记录每个Pair的频数
            token_pair_map: Dict[Tuple[bytes, bytes], int] = {}
            for pre_token in pre_tokenization_map:
                if len(pre_token) == 1:
                    continue

                count = pre_tokenization_map[pre_token]
                for token_pair in zip(pre_token[:-1], pre_token[1:]):
                    if token_pair in token_pair_map:
                        token_pair_map[token_pair] += count
                    else:
                        token_pair_map[token_pair] = count

            # 找到其中频数最大的一个Pair
            # 如果所有pre token都只有一个bytes了，没法组成pair，那也结束
            if len(token_pair_map) == 0:
                break
            count, token_pair = max(((token_pair_map[token_pair], token_pair) for token_pair in token_pair_map))
            # 合并这个Pair
            new_token = token_pair[0] + token_pair[1]
            # 添加到记录里
            self.vocab[len(self.vocab)] = new_token
            self.merges.append(token_pair)
            # 更新pre token map
            def replace_target_pair(source: Tuple[bytes, ...], target: Tuple[bytes, bytes]) -> Tuple[bytes, ...]:
                assert len(target) == 2
                if len(source) < 2:
                    return source

                target_combined = target[0] + target[1]
                result: List[bytes] = []
                i = 0
                while i < len(source):
                    if i < len(source) - 1 and source[i] == target[0] and source[i + 1] == target[1]:
                        result.append(target_combined)
                        i += 2
                    else:
                        result.append(source[i])
                        i += 1
                return tuple(result)

            new_pre_token_map = {}
            for pre_token in pre_tokenization_map:
                old_count = pre_tokenization_map[pre_token]
                new_pre_token = replace_target_pair(pre_token, token_pair)
                new_pre_token_map[new_pre_token] = old_count
            pre_tokenization_map = new_pre_token_map

        return self

    def encode(self, text: str) -> List[int]:
        pre_tokenization_result: List[Tuple[bytes, ...]] = self._pre_tokenization(text, keep_special_tokens=True)

        # pre_tokenization_result里面，bytes的是普通token，str的是特殊token
        reverse_vocab = {self.vocab[key]: key for key in self.vocab}
        special_token_vocab_map = {token: reverse_vocab[token.encode()] for token in self.special_tokens}
        merge_rank_map: Dict[Tuple[bytes, bytes], int] = {merge: i for i, merge in enumerate(self.merges)}
        encode_result: List[int] = []

        for pre_token in pre_tokenization_result:
            if pre_token in special_token_vocab_map:
                encode_result.append(special_token_vocab_map[pre_token])
                continue
            if pre_token in self.pre_token_encode_result_cache:
                encode_result.extend(self.pre_token_encode_result_cache[pre_token])
                continue

            pre_token_encode_result = self._encode_one_pre_token(pre_token, reverse_vocab, merge_rank_map)
            encode_result.extend(pre_token_encode_result)
            self.pre_token_encode_result_cache[pre_token] = pre_token_encode_result

        return encode_result

    def _pre_tokenization(self, text, keep_special_tokens: bool) -> List[Tuple[bytes, ...]]:
        pre_tokenization_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # 这里让更长的special token在更前面
        if len(self.special_tokens) == 0:
            parts = [text]
        else:
            special_token_regex = "|".join([re.escape(special_token) for special_token in sorted(self.special_tokens, key=lambda t: -len(t))])
            if keep_special_tokens:
                special_token_regex = "(" + special_token_regex + ")"
            parts = re.split(special_token_regex, text)

        pre_tokenization_result: List[Tuple[bytes, ...]] = []
        for part in parts:
            if part in self.special_tokens:
                pre_tokenization_result.append((part.encode(), ))
                continue

            pre_tokens: List[str] = re.findall(pre_tokenization_pat, part)
            for token_str in pre_tokens:
                pre_tokenization_result.append(tuple((bytes([b]) for b in token_str.encode())))
        return pre_tokenization_result

    def _encode_one_pre_token(self, token: Tuple[bytes, ...], reverse_vocab: Dict[bytes, int], merge_rank_map: Dict[Tuple[bytes, bytes], int]) -> List[int]:
        assert len(token) > 0
        if len(token) == 1:
            return [reverse_vocab[token[0]]]

        class BytesNode:
            def __init__(self, content: bytes, prev: Optional["BytesNode"], next: Optional["BytesNode"]):
                self.content = content
                self.prev = prev
                self.next = next
            def __repr__(self):
                return self.content.__repr__()

        class BytesPair:
            def __init__(self, rank: int, pair: Tuple[BytesNode, BytesNode]):
                self.rank = rank
                self.pair = pair
                self.valid = True

            def __lt__(self, other):
                return self.rank < other.rank

        # 建立Merge链表
        dummy_head: BytesNode = BytesNode(bytes(), None, None)
        current_node: BytesNode = dummy_head
        for content in token:
            next_node = BytesNode(content, current_node, None)
            current_node.next = next_node
            current_node = next_node

        # 建立初始堆
        # (Pair, Valid)
        pair_heap: List[BytesPair] = []
        pair_heap_map: Dict[Tuple[bytes, bytes], BytesPair] = {}

        current_node = dummy_head.next
        while True:
            if current_node.next is None:
                break
            node_pair = (current_node, current_node.next)
            bytes_pair = (current_node.content, current_node.next.content)
            if bytes_pair in merge_rank_map:
                rank = merge_rank_map[bytes_pair]
                pair = BytesPair(rank, node_pair)
                pair_heap.append(pair)
                pair_heap_map[bytes_pair] = pair

            current_node = current_node.next
        heapq.heapify(pair_heap)

        # 持续合并
        while len(pair_heap) != 0:
            pair = pair_heap[0]
            heapq.heappop(pair_heap)
            if not pair.valid:
                continue
            # print(f"Merge {pair.pair}")
            # 合并这个Pair，然后把新的Pair添加进堆
            # 如果这个Pair是(b, c)，那么要把前面的(a, b)和后面的(c, d)无效掉
            # 然后添加(a, bc) 和 (bc, d)
            b, c = pair.pair
            bc = BytesNode(b.content + c.content, None, None)
            if b.prev is not None:
                a = b.prev
                # 处理链表
                a.next = bc
                bc.prev = a
                # 无效化(a, b)
                if (a.content, b.content) in pair_heap_map:
                    pair_heap_map[(a.content, b.content)].valid = False
                # 添加新的Pair（如果这个Pair合法的话）
                new_bytes_pair = (a.content, bc.content)
                if new_bytes_pair in merge_rank_map:
                    new_rank = merge_rank_map[new_bytes_pair]
                    new_node_pair = (a, bc)
                    new_pair = BytesPair(new_rank, new_node_pair)
                    heapq.heappush(pair_heap, new_pair)
                    pair_heap_map[new_bytes_pair] = new_pair

            if c.next is not None:
                d = c.next
                # 处理链表
                d.prev = bc
                bc.next = d
                # 无效化(c, d)
                if (c.content, d.content) in pair_heap_map:
                    pair_heap_map[(c.content, d.content)].valid = False
                # 添加新的Pair（如果这个Pair合法的话）
                new_bytes_pair = (bc.content, d.content)
                if new_bytes_pair in merge_rank_map:
                    new_rank = merge_rank_map[new_bytes_pair]
                    new_node_pair = (bc, d)
                    new_pair = BytesPair(new_rank, new_node_pair)
                    heapq.heappush(pair_heap, new_pair)
                    pair_heap_map[new_bytes_pair] = new_pair

        # 遍历链表，的到最终的encode结果
        current_node = dummy_head.next
        encode_result: List[int] = []
        while current_node is not None:
            encode_result.append(reverse_vocab[current_node.content])
            current_node = current_node.next

        return encode_result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            encode_result = self.encode(text)
            for token_id in encode_result:
                yield token_id


    def decode(self, token_ids: List[int]) -> str:
        decode_result: List[bytes] = []
        for token_id in token_ids:
            byte = self.vocab[token_id]
            decode_result.append(byte)
        return b"".join(decode_result).decode(errors="replace")

    def dump(self, dir: str | os.PathLike):
        os.makedirs(dir, exist_ok=True)

        with open(os.path.join(dir, "vocab.pkl"), "wb") as vocab_file, open(os.path.join(dir, "merges.pkl"), "wb") as merges_file, \
            open(os.path.join(dir, "special_tokens.txt"), "w", encoding="utf-8") as special_token_file:
            pickle.dump(self.vocab, vocab_file)
            pickle.dump(self.merges, merges_file)
            special_token_file.write("\n".join(self.special_tokens))


    @classmethod
    def from_files(cls, tokenizer_dir: str, special_tokens: Optional[List[str]]=None):
        tokenizer = cls(None, None, None)
        tokenizer_special_tokens: List[str] = []

        vocab_filepath = os.path.join(tokenizer_dir, "vocab.pkl")
        merges_filepath = os.path.join(tokenizer_dir, "merges.pkl")
        special_token_filepath = os.path.join(tokenizer_dir, "special_tokens.txt")
        with open(vocab_filepath, "rb") as vocab_file, open(merges_filepath, "rb") as merges_file, open(special_token_filepath, "r") as special_token_file:
            tokenizer.vocab = pickle.load(vocab_file)
            tokenizer.merges = pickle.load(merges_file)
            tokenizer_special_tokens = [token for token in special_token_file.read().split("\n") if token]

        tokenizer._append_user_special_tokens(tokenizer_special_tokens)
        tokenizer._append_user_special_tokens(special_tokens)

        return tokenizer


if __name__ == '__main__':
    # tokenizer = BPETokenizer.train_bpe("/home/qzero/cs336/assignment1-basics/data/simple_text.txt", 6+256, ["<|endoftext|>"])
    # print(tokenizer.vocab)
    # print(tokenizer.merges)
    # print(tokenizer.special_tokens)

    # reverse_vocab = {tokenizer.vocab[key]: key for key in tokenizer.vocab}
    # merge_rank_map: Dict[Tuple[bytes, bytes], int] = {merge: i for i, merge in enumerate(tokenizer.merges)}
    # token = tuple((bytes([b]) for b in "lowest".encode()))
    # print(tokenizer._encode_one_pre_token(token, reverse_vocab, merge_rank_map))
    # print(tokenizer.encode("the cat ate"))
    # tokenizer.train_bpe("/home/qzero/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    # tokenizer.train_bpe("/home/qzero/cs336/assignment1-basics/data/simple_text.txt", 20+256, ["<|endoftext|>"])
    # tokenizer.dump("/home/qzero/cs336/assignment1-basics/data/saved_tokenizer_story/")
    # new_tokenizer = BPETokenizer.from_files("/home/qzero/cs336/assignment1-basics/data/saved_tokenizer_story/")
    # print(new_tokenizer.vocab)
    # print(new_tokenizer.merges)
    # assert tokenizer.vocab == new_tokenizer.vocab
    # assert tokenizer.merges == new_tokenizer.merges

    tokenizer = BPETokenizer.from_files("/home/qzero/cs336/assignment1-basics/data/saved_tokenizer_story/vocab.pkl",
                                        "/home/qzero/cs336/assignment1-basics/data/saved_tokenizer_story/merges.pkl",
                                        ["<|endoftext|>"])
    # ids = tokenizer.encode("For example, suppose our input string is<|endoftext|>some how wonderful")
    # print(ids)
    # print(tokenizer.decode([6795, 8713, 45, 2367, 2296, 1144, 313, 113, 321, 215]))
    # tokenizer.dump("/home/qzero/cs336/assignment1-basics/data/saved_tokenizer_story/")
    # with open("/home/qzero/cs336/assignment1-basics/test/fixtures/tinystories_sample_5M.txt") as f:
    #     ids = []
    #     for _id in tokenizer.encode_iterable(f):
    #         ids.append(_id)
    #     print(ids)
