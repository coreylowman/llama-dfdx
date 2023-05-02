use rand::{rngs::StdRng, Rng};

pub fn greedy(probs: Vec<f32>) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .map(|x| x.0)
        .unwrap()
}

pub fn top_p(probs: Vec<f32>, top_p: f32, top_k: usize, rng: &mut StdRng) -> usize {
    // Add in index alongsize probability
    let mut probs_sort: Vec<(usize, f32)> = probs.into_iter().enumerate().collect::<Vec<_>>();

    // NOTE: descending
    probs_sort.sort_unstable_by(|&(_, a), &(_, b)| b.total_cmp(&a));

    // Only keep first top_k values
    let mut n_choices = top_k;

    // Only keep first n_choices values,
    // where `sum(probs_sort[..n_choices]) >= top_p`
    let mut total = 0.0;
    for &(i, prob) in probs_sort.iter().take(n_choices) {
        total += prob;
        if total >= top_p {
            n_choices = i + 1;
            break;
        }
    }

    // Sampling from the remaining choices.
    // This is equivalent to sampling from rand::WeightedIndex
    // and also sampling from a multinomial distribution.
    let p: f32 = rng.gen_range(0.0..total);
    let mut total = 0.0;
    for &(i, prob) in probs_sort.iter().take(n_choices) {
        total += prob;
        if total >= p {
            return i;
        }
    }

    unreachable!()
}
