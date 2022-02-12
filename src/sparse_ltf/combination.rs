pub(super) struct CombIter {
    max:    usize,
    first:  usize,
    second: usize,
}


impl CombIter {
    pub(super) fn new(max: usize) -> CombIter {
        CombIter {
            max:    max,
            first:    0,
            second:   0,
        }
    }
}


impl Iterator for CombIter {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        if self.first >= self.max {
            return None;
        }

        if self.second >= self.max - 1 {
            self.first += 1;
            self.second = self.first + 1;
        } else {
            self.second += 1;
        }

        if self.first == self.max - 1 {
            None
        } else {
            Some((self.first, self.second))
        }
    }
}
