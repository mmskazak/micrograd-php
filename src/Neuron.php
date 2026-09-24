<?php

declare(strict_types=1);

/**
 * Нейрон: out = tanh(b + w1*x1 + w2*x2 + ... + wn*xn)
 */
final class Neuron
{
    /** @var Value[] */
    public array $w = [];
    public Value $b;

    /**
     * Промежуточные узлы последнего прямого прохода — только для визуализации графа.
     * ['x' => Value[], 'muls' => Value[], 'sums' => Value[], 'out' => Value]
     */
    public array $trace = [];

    public function __construct(int $nin)
    {
        // случайно в [−1, 1], как в видео, но поделено на √nin: чем больше входов,
        // тем меньше каждый вес, чтобы сумма w·x не загоняла tanh в насыщение (±1),
        // где его производная ≈ 0 и нейрон перестаёт учиться
        $scale = 1 / sqrt($nin);
        for ($i = 0; $i < $nin; $i++) {
            $this->w[] = new Value((mt_rand() / mt_getrandmax() * 2 - 1) * $scale, [], '', 'w' . ($i + 1));
        }
        $this->b = new Value(0.0, [], '', 'b');
    }

    /** @param Value[] $x */
    public function __invoke(array $x): Value
    {
        $act = $this->b;
        $muls = [];
        $sums = [];
        for ($i = 0; $i < count($this->w); $i++) {
            $m = $this->w[$i]->mul($x[$i]);
            $act = $act->add($m);
            $muls[] = $m;
            $sums[] = $act;
        }
        $out = $act->tanh();

        $this->trace = ['x' => $x, 'muls' => $muls, 'sums' => $sums, 'out' => $out];
        return $out;
    }

    /** @return Value[] */
    public function parameters(): array
    {
        return [...$this->w, $this->b];
    }
}
