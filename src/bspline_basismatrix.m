function B = bspline_basismatrix(degree, knots, x)
    n = length(x);
    k = degree + 1;
    m = length(knots) - k;
    B = zeros(n, m);
    for i = 1:m
        B(:, i) = bspline_basis(i, degree, knots, x);
    end
end

function y = bspline_basis(i, k, t, x)
    if k == 0
        y = double(t(i) <= x & x < t(i+1));
    else
        left = 0; right = 0;
        if t(i+k) ~= t(i)
            left = (x - t(i)) ./ (t(i+k) - t(i)) .* bspline_basis(i, k-1, t, x);
        end
        if t(i+k+1) ~= t(i+1)
            right = (t(i+k+1) - x) ./ (t(i+k+1) - t(i+1)) .* bspline_basis(i+1, k-1, t, x);
        end
        y = left + right;
    end
end
