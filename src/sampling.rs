use rand::rngs::StdRng;

pub trait TokenSampler {
    fn sample(&self, probs: Vec<f32>) -> usize;
}

pub struct GreedySampler;

impl TokenSampler for GreedySampler {
    fn sample(&self, probs: Vec<f32>) -> usize {
        probs
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
            .map(|x| x.0)
            .unwrap()
    }
}

pub struct TopPSampler {
    top_p: f32,
    temperature: f32,
    rng: StdRng,
}

impl TokenSampler for TopPSampler {
    fn sample(&self, probs: Vec<f32>) -> usize {
        let mut probs_sort = probs.as_vec().into_iter().enumerate().collect::<Vec<_>>();
        probs_sort.sort_unstable_by(|&(_, a), &(_, b)| b.total_cmp(&a)); // NOTE: descending
        let mut total = 0.0;
        let mut n_choices = modeling::VOCAB;
        for i in 0..n_choices {
            total += probs_sort[i].1;
            if total >= args.top_p {
                n_choices = i + 1;
                break;
            }
        }
        let p: f32 = rng.gen_range(0.0..total);
        for i in 0..n_choices {
            if probs_sort[i].1 >= p {}
        }
    }
}
