function idx = discretize_state(state, n_bins, bounds)
    idx = zeros(1, 4);
    for i = 1:4
        low = bounds(i,1);
        high = bounds(i,2);
        value = state(i);
        value = min(max(value, low), high); % klipp inom gränser
        idx(i) = floor(n_bins * (value - low) / (high - low)) + 1;
        idx(i) = min(max(idx(i), 1), n_bins);
    end
end
