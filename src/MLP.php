<?php

declare(strict_types=1);

final class MLP
{
    /** @var int[] например [9, 12, 10] */
    public array $sizes;
    /** @var Layer[] */
    public array $layers;

    public function __construct(int $nin, array $nouts)
    {
        $this->sizes = [$nin, ...$nouts];
        $this->layers = [];
        for ($i = 0; $i < count($nouts); $i++) {
            $this->layers[] = new Layer($this->sizes[$i], $this->sizes[$i + 1]);
        }
    }

    /**
     * @param array<float|Value> $x
     * @return Value[] выходы последнего слоя
     */
    public function __invoke(array $x): array
    {
        $inputs = [];
        foreach ($x as $i => $xi) {
            $inputs[] = $xi instanceof Value ? $xi : new Value((float)$xi, [], '', 'x' . ($i + 1));
        }

        $x = $inputs;
        foreach ($this->layers as $layer) {
            $x = $layer($x); // выход слоя становится входом следующего
        }
        return $x;
    }

    /** @return Value[] */
    public function parameters(): array
    {
        $params = [];
        foreach ($this->layers as $layer) {
            array_push($params, ...$layer->parameters());
        }
        return $params;
    }
}
