from .llvm import BasicBlock, IcmpOp, Phi, Value

class ForLoopBuilder:
    _entry: BasicBlock
    _step: Value
    _phi: Phi
    _comp: BasicBlock
    _body_entry: BasicBlock
    _output_block: BasicBlock

    def __init__(self, entry: BasicBlock, signed: bool, lower: Value, upper: Value, step: Value) -> None:
        self._entry = entry
        self._comp = BasicBlock()
        self._body_entry = BasicBlock()
        self._output_block = BasicBlock()
        self._step = step

        entry.jmp(self._comp)
        self._phi = self._comp.phi((lower, entry))
        self._comp.br(
            self._comp.icmp(IcmpOp.LE, signed, self._phi, upper),
            self._body_entry,
            self._output_block,
        )

    def end(self, body_end: BasicBlock):
        next_var = body_end.add(self._phi, self._step)
        body_end.jmp(self._comp)
        self._phi.add_incoming(next_var, body_end)
        return self._output_block

    @property
    def loop_var(self):
        return self._phi

    @property
    def body_entry(self):
        return self._body_entry
