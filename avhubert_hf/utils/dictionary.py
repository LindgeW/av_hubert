import os
from typing import Dict, List, Optional, Set


class Dictionary:
    """A mapping from symbols to consecutive integers"""
    
    def __init__(
        self,
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        bos="<s>",
        extra_special_symbols=None,
    ):
        self.pad_word = pad
        self.eos_word = eos
        self.unk_word = unk
        self.bos_word = bos
        
        self.symbols = []
        self.count = []
        self.indices = {}
        
        # Add special symbols
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        self.bos_index = self.add_symbol(bos)
        
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        
        self.nspecial = len(self.symbols)
    
    def __eq__(self, other):
        return self.indices == other.indices
    
    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word
    
    def __len__(self):
        return len(self.symbols)
    
    def __contains__(self, sym):
        return sym in self.indices
    
    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index
    
    def string(self, tensor, bpe_symbol=None, escape_unk=False, extra_symbols_to_ignore=None, unk_string=None):
        """Helper for converting a tensor of token indices to a string.
        
        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(self.string(t, bpe_symbol, escape_unk, extra_symbols_to_ignore) for t in tensor)
        
        def token_string(i):
            if i == self.unk():
                if unk_string is not None:
                    return unk_string
                else:
                    return self.unk_word
            else:
                return self[i]
        
        sent = " ".join(
            token_string(i)
            for i in tensor
            if (extra_symbols_to_ignore is None or i not in extra_symbols_to_ignore)
        )
        
        return sent
    
    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx
    
    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word, count in new_dict.count.items():
            idx2count = new_dict.count[word]
            idx = self.indices.get(word, None)
            if idx is None:
                idx = self.add_symbol(word)
                self.count[idx] = idx2count
            else:
                self.count[idx] += idx2count
    
    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.
        
        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
              including special symbols. -1 means use all words.
            - padding_factor can be used to pad the dictionary size to be a
              multiple of 8, which is important on some hardware (e.g., Nvidia
              Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)
        
        new_indices = dict(zip(self.symbols[:self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[:self.nspecial]
        new_count = self.count[:self.nspecial]
        
        c = sorted(
            zip(self.symbols[self.nspecial:], self.count[self.nspecial:]),
            key=lambda x: (x[1], x[0]),
            reverse=True,
        )
        
        for i, (sym, cnt) in enumerate(c):
            if cnt >= threshold:
                new_indices[sym] = len(new_symbols)
                new_symbols.append(sym)
                new_count.append(cnt)
                if len(new_symbols) >= nwords:
                    break
        
        assert len(new_symbols) == len(new_indices)
        
        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices
    
    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index
    
    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index
    
    def unk(self):
        """Helper to get index of unknown symbol"""
        return self.unk_index
    
    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index
    
    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the following format:
        
        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d
    
    def add_from_file(self, f):
        """Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return
        
        lines = f.readlines()
        indices_start_line = self._load_meta(lines)
        
        for line in lines[indices_start_line:]:
            try:
                line, field = line.rstrip().rsplit(" ", 1)
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: "
                        "{}. Please duplicate words in the dictionary file, "
                        "or set the overwrite flag. ".format(word)
                    )
                self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    f"Incorrect dictionary format, expected '<token> <count>', "
                    f"got line: {line}"
                )
    
    def _load_meta(self, lines):
        return 0
    
    def save(self, f):
        """Stores dictionary into a text file"""
        if isinstance(f, str):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, "w", encoding="utf-8") as fd:
                return self.save(fd)
        
        for symbol, count in zip(self.symbols, self.count):
            print("{} {}".format(symbol, count), file=f)
    
    def dummy_sentence(self, length):
        """Constructs a dummy sentence of a given length."""
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t
    
    def encode_line(
        self,
        line,
        line_tokenizer=None,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
    ) -> torch.IntTensor:
        """Encode a single line into a tensor of indices.
        
        Args:
            line: a single line to encode
            line_tokenizer: tokenizer to use for the line
            add_if_not_exist: if True, adds new tokens to the dictionary
            consumer: if not None, calls consumer(word) for each word
            append_eos: if True, appends end-of-sentence symbol
            reverse_order: if True, reverses the order of tokens
        """
        words = line_tokenizer(line) if line_tokenizer else line.split()
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)
        
        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word)
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids


# Import torch here to avoid circular imports
import torch