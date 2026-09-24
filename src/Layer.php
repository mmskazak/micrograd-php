<?php

declare(strict_types=1);

final class Layer
{
    /** @var Neuron[] */
    public array $neurons;

    public function __construct(int $nin, int $nout)
    {
        $this->neurons = [];
        for ($i = 0; $i < $nout; $i++) {
            $this->neurons[] = new Neuron($nin);
        }
    }

    /**
     * @param Value[] $x
     * @return Value[]
     */
    public function __invoke(array $x): array
    {
        $outs = [];
        foreach ($this->neurons as $neuron) {
            $outs[] = $neuron($x);
        }
        return $outs;
    }

    /** @return Value[] */
    public function parameters(): array
    {
        $params = [];
        foreach ($this->neurons as $neuron) {
            array_push($params, ...$neuron->parameters());
        }
        return $params;
    }
}
