<?php

declare(strict_types=1);

/**
 * Value — атомарное "умное число" из micrograd.
 *
 * data  — само значение.
 * grad  — насколько изменится итоговый loss при малом изменении этого узла (∂loss/∂this).
 * prev  — родители в графе вычислений (из кого этот узел получен).
 * op    — какой операцией получен ('+', '*', 'tanh' или '' для листа).
 *
 * _backward только ЧИТАЕТ out->grad и НАКАПЛИВАЕТ (+=) grad родителей.
 * data он никогда не трогает — обновление весов делается отдельным шагом.
 */
final class Value
{
    public float $data;
    public float $grad = 0.0;
    /** @var Value[] */
    public array $prev;
    public string $op;
    public string $label;
    private ?Closure $backwardFn = null;

    public function __construct(float $data, array $children = [], string $op = '', string $label = '')
    {
        $this->data = $data;
        $this->prev = $children;
        $this->op = $op;
        $this->label = $label;
    }

    public static function wrap(Value|float|int $x): Value
    {
        return $x instanceof Value ? $x : new Value((float)$x);
    }

    public function add(Value|float|int $other): Value
    {
        $other = self::wrap($other);
        $out = new Value($this->data + $other->data, [$this, $other], '+');
        $out->backwardFn = function () use ($out, $other): void {
            // производная суммы по каждому слагаемому = 1: градиент просто "протекает"
            $this->grad += $out->grad;
            $other->grad += $out->grad;
        };
        return $out;
    }

    public function mul(Value|float|int $other): Value
    {
        $other = self::wrap($other);
        $out = new Value($this->data * $other->data, [$this, $other], '*');
        $out->backwardFn = function () use ($out, $other): void {
            // производная a*b по a = b, по b = a
            $this->grad += $other->data * $out->grad;
            $other->grad += $this->data * $out->grad;
        };
        return $out;
    }

    public function sub(Value|float|int $other): Value
    {
        return $this->add(self::wrap($other)->mul(-1));
    }

    public function tanh(): Value
    {
        $t = tanh($this->data);
        $out = new Value($t, [$this], 'tanh');
        $out->backwardFn = function () use ($out, $t): void {
            // d tanh(x) / dx = 1 - tanh(x)^2
            $this->grad += (1 - $t * $t) * $out->grad;
        };
        return $out;
    }

    public function backward(): void
    {
        // топологический порядок: каждый узел идёт после всех своих родителей
        $topo = [];
        $visited = [];
        $build = function (Value $v) use (&$build, &$topo, &$visited): void {
            $id = spl_object_id($v);
            if (isset($visited[$id])) {
                return;
            }
            $visited[$id] = true;
            foreach ($v->prev as $child) {
                $build($child);
            }
            $topo[] = $v;
        };
        $build($this);

        $this->grad = 1.0;
        for ($i = count($topo) - 1; $i >= 0; $i--) {
            $node = $topo[$i];
            if ($node->backwardFn !== null) {
                ($node->backwardFn)();
            }
        }
    }
}
