<?php

declare(strict_types=1);

/**
 * Один шаг обучения = одна эпоха по всем 10 цифрам:
 * forward → loss → zero_grad → backward → update.
 */
final class Trainer
{
    /**
     * loss = (1/N) · Σ_примеров Σ_выходов (ypred − y)²   — среднее по примерам, как в конспекте
     *
     * @return array{loss: Value, preds: array<int, Value[]>}
     */
    public static function loss(MLP $model, array $digits): array
    {
        $loss = new Value(0.0);
        $preds = [];
        foreach ($digits as $digit) {
            $ypred = $model(Digits::encode(Digits::toInput($digit)));
            $ys = Digits::target($digit);
            foreach ($ypred as $k => $yk) {
                $diff = $yk->sub($ys[$k]);
                $loss = $loss->add($diff->mul($diff));
            }
            $preds[$digit] = $ypred;
        }
        $loss = $loss->mul(1 / count($digits));
        $loss->label = 'loss';
        return ['loss' => $loss, 'preds' => $preds];
    }

    /** @return array{loss: float, accuracy: float} */
    public static function epoch(MLP $model, float $lr): array
    {
        // 1–2. forward + loss
        ['loss' => $loss, 'preds' => $preds] = self::loss($model, array_keys(Digits::DIGITS));

        // 3. zero_grad
        foreach ($model->parameters() as $p) {
            $p->grad = 0.0;
        }

        // 4. backward
        $loss->backward();

        // 5. update
        foreach ($model->parameters() as $p) {
            $p->data -= $lr * $p->grad;
        }

        $correct = 0;
        foreach ($preds as $digit => $ypred) {
            if (self::argmax($ypred) === $digit) {
                $correct++;
            }
        }
        return ['loss' => $loss->data, 'accuracy' => $correct / count($preds)];
    }

    /** @param Value[] $outs */
    public static function argmax(array $outs): int
    {
        $best = 0;
        foreach ($outs as $k => $v) {
            if ($v->data > $outs[$best]->data) {
                $best = $k;
            }
        }
        return $best;
    }
}
