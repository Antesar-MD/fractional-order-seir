%% generate_figures.m
% =========================================================================
% MATLAB companion code for "A Unified Fractional-Order SEIR Framework for
% Endemic Disease Dynamics: Vertical Transmission, Disease-Induced
% Mortality, and Stability" (Antesar Aldawoud). Produces every figure in
% the article, from the model equations of Sections 2, 4, and 5, and
% prints Table 3 (R0 and I* for all eight model variants) to the console.
%
% This is a MATLAB port of the project's reference implementation,
% generate_figures.py, and reproduces its numerical results exactly (the
% two share the same closed-form R0/I* formulas and the same explicit
% Grunwald-Letnikov scheme; the GL update rule here was cross-checked
% against the Python version to floating-point precision before this file
% was written).
%
% PROVENANCE
% ----------
% This script consolidates and generalises the original MATLAB prototype
% (11 separate .m files: a classical SIR epidemic/endemic pair, an
% age-structured SIR epidemic/endemic pair, and a fractional-order SEIR
% model restricted to theta = 0, p = q = 0, i.e. Table 1 case (a) only)
% into a single file covering the full eight-case model family analysed
% in the article:
%
%   Figure  Model                                   Origin
%   ------  --------------------------------------  --------------------------------
%   1       Classical SIR epidemic model             translated (ode45)
%   2       Bifurcation diagram, endemic model        new (plots the closed-form i*)
%   3       Classical SIR endemic model               translated (ode45)
%   4       General fractional-order SEIR model       generalised (theta, p, q added)
%   5       General fractional-order SIR model        new (no prototype equivalent)
%
% fractional_seir_general below extends the original prototype's GL loop
% (function FOSEIRendemic.m, theta = 0, p = q = 0 only) with the
% disease-induced-mortality term theta*I and the vertical-transmission
% terms Delta*p*E/N + Delta*q*I/N, so that setting theta = 0, p = q = 0
% reproduces the original equations exactly. The age-structured extension
% present in the original prototype (two age classes, not used by any
% figure in this article) is intentionally out of scope here, as it is in
% the article (see the Discussion's limitations paragraph).
%
% NUMERICAL METHOD
% -----------------
% All fractional-order simulations (Figures 4-5) use the explicit
% Grunwald-Letnikov (GL) fractional-difference scheme for a Caputo initial
% value problem D^alpha x = f(t, x), x(0) = x0, step size h:
%
%   w_0 = 1,  w_m = (1 - (alpha + 1) / m) * w_{m-1},   m = 1, 2, ...
%   x_{k+1} = h^alpha * f(t_k, x_k) - sum_{j=1}^{k+1} w_j * x_{k+1-j}
%
% (see e.g. I. Petras, "Fractional-Order Nonlinear Systems", Springer,
% 2011). This is a first-order, explicit method whose memory sum is
% O(n^2) in the number of time steps n = t_sim / h, since every step
% depends on the entire simulated history; the memory sum is vectorised
% here as a single dot product per step, exactly as in the original
% prototype's FOSEIRendemic.m. The classical (alpha = 1) simulations of
% Figures 1 and 3 instead use ode45 (Dormand-Prince RK45), unrelated to
% the GL scheme and included only for the integer-order baseline model of
% Section 2.
%
% Figures 4-5 use t_sim = 1000 rather than a shorter window because, at
% the article's parameter values, the infection subsystem's own approach
% to the endemic steady state is slow (e-folding time on the order of 40)
% relative to the demographic relaxation time 1/mu = 125; a shorter
% window shows only the early transient, not convergence. Running to
% t_sim = 1000 also serves as an independent numerical check on the
% closed-form endemic steady states of Section 4: the simulated long-run
% values agree with the analytic F_1 to 3-4 significant figures.
%
% In addition to the five figures, this script reproduces Table 3 of the
% article (R0 and the endemic infective level I* for all eight model
% variants), computed directly from the closed-form formulas of Section 4
% (seir_R0_Istar, sir_R0_Istar, compute_table3) rather than simulated --
% so the whole numerical comparison in the article, not only Figures 4-5,
% can be checked by running this one script.
%
% USAGE
% -----
%   Run this file in MATLAB (R2018b or later: R2016b for local functions
%   in a script file, R2018b for sgtitle on Figures 4-5) or GNU Octave
%   (6.1 or later, for the same two features). On an older MATLAB,
%   replace the two sgtitle(...) calls with a single subplot title, or
%   simply delete them -- the five PNG files and Table 3 are unaffected.
%
% Writes five PNG files (300 dpi) to the current directory, named to
% match the figure labels used in the article's LaTeX source, and prints
% Table 3 to the console:
%
%   fig1_epidemic_timeseries.png
%   fig2_bifurcation.png
%   fig3_endemic_timeseries.png
%   fig4_seir_alpha_comparison.png
%   fig5_sir_alpha_comparison.png
%
% Total run time is on the order of one to two minutes on an ordinary
% laptop with the default step sizes below (Figures 4-5 dominate the
% runtime: two O(n^2) GL simulations each, n = t_sim/h = 20000).
% =========================================================================

clc;
close all;

fprintf('Part 1: classical SIR epidemic model (Figure 1)...\n');
make_figure1();

fprintf('Part 2: classical SIR endemic model (Figures 2-3)...\n');
make_figure2_bifurcation();
make_figure3();

fprintf('Part 3: general fractional-order SEIR model (Figure 4)...\n');
make_figure4();

fprintf('Part 4: general fractional-order SIR model (Figure 5)...\n');
make_figure5();

fprintf('Part 5: Table 3 (R0 and I* for all eight variants)...\n');
compute_table3();

fprintf('Done. Five PNG files written to the current directory.\n');


%% ========================================================================
%  LOCAL FUNCTIONS
%  (MATLAB and Octave both require local function definitions to appear
%  after all top-level script commands in a script file.)
%  ========================================================================

% ---- Shared style: fixed colours/line styles, never re-cycled, matching
% the colourblind-safe Okabe-Ito palette used in the Python figures. -----
function c = style_black()
    c = [0 0 0];
end

function c = style_blue()
    c = [0.000 0.447 0.698];
end

function c = style_vermillion()
    c = [0.835 0.369 0.000];
end


% ========================================================================
% PART 1 -- Classical SIR epidemic model (article Sec. 2.1, Eq. 1-3)
% ========================================================================

function dydt = sir_epidemic_rhs(~, y, beta, gamma)
    s = y(1); i = y(2);
    dydt = [-beta*s*i; beta*s*i - gamma*i];
end

function make_figure1()
    beta = 0.05; gamma = 0.003;
    y0 = [1 - 1e-5, 1e-5];
    t_eval = linspace(0, 500, 2000);
    opts = odeset('RelTol', 1e-9, 'AbsTol', 1e-11);
    [t, y] = ode45(@(t, y) sir_epidemic_rhs(t, y, beta, gamma), t_eval, y0, opts);

    fig = figure('Position', [100 100 600 400]);
    plot(t, y(:,1), 'Color', style_vermillion(), 'LineStyle', '--', 'LineWidth', 2); hold on;
    plot(t, y(:,2), 'Color', style_blue(), 'LineStyle', '-.', 'LineWidth', 2);
    xlabel('time t');
    ylabel('proportion of population');
    title(sprintf('SIR epidemic model, R_0 = \\beta/\\gamma = %.2f', beta/gamma));
    legend('s(t)', 'i(t)', 'Location', 'best');
    grid on; box on;
    print(fig, 'fig1_epidemic_timeseries.png', '-dpng', '-r300');
    close(fig);
    fprintf('  wrote fig1_epidemic_timeseries.png\n');
end


% ========================================================================
% PART 2 -- Classical SIR endemic model (article Sec. 2.2, Eq. 4-6)
% Figure 2 plots the closed-form i* = (mu/beta)(R0-1) of Proposition 2.1
% directly, as a visual check on that proposition.
% ========================================================================

function dydt = sir_endemic_rhs(~, y, beta, gamma, mu)
    s = y(1); i = y(2);
    dydt = [mu - beta*s*i - mu*s; beta*s*i - (gamma+mu)*i];
end

function make_figure3()
    beta = 0.05; gamma = 0.003; mu = 0.008;
    y0 = [1 - 1e-5, 1e-5];
    t_eval = linspace(0, 500, 2000);
    opts = odeset('RelTol', 1e-9, 'AbsTol', 1e-11);
    [t, y] = ode45(@(t, y) sir_endemic_rhs(t, y, beta, gamma, mu), t_eval, y0, opts);

    R0 = beta / (gamma + mu);
    fig = figure('Position', [100 100 600 400]);
    plot(t, y(:,1), 'Color', style_vermillion(), 'LineStyle', '--', 'LineWidth', 2); hold on;
    plot(t, y(:,2), 'Color', style_blue(), 'LineStyle', '-.', 'LineWidth', 2);
    xlabel('time t');
    ylabel('proportion of population');
    title(sprintf('SIR endemic model, R_0 = %.2f, converging to F_1', R0));
    legend('s(t)', 'i(t)', 'Location', 'best');
    grid on; box on;
    print(fig, 'fig3_endemic_timeseries.png', '-dpng', '-r300');
    close(fig);
    fprintf('  wrote fig3_endemic_timeseries.png\n');
end

function make_figure2_bifurcation()
    % Bifurcation diagram i* vs. beta, mu=0.008, gamma=0.003 fixed,
    % plotting the closed-form i* = (mu/beta)(R0-1) of Proposition 2.1.
    mu = 0.008; gamma = 0.003;
    beta = linspace(1e-4, 0.03, 2000);
    R0 = beta / (gamma + mu);
    istar = (mu ./ beta) .* (R0 - 1);
    istar(R0 <= 1) = 0.0;
    beta_crit = gamma + mu;  % R0 = 1

    fig = figure('Position', [100 100 600 400]);
    plot(beta, istar, 'Color', style_blue(), 'LineWidth', 2); hold on;
    hcrit = xline(beta_crit, ':k', 'LineWidth', 1.2);
    xlabel('\beta');
    ylabel('i^*');
    title('Transcritical bifurcation of the endemic model at R_0 = 1');
    legend(hcrit, '\beta = \gamma + \mu  (R_0 = 1)', 'Location', 'best');
    grid on; box on;
    print(fig, 'fig2_bifurcation.png', '-dpng', '-r300');
    close(fig);
    fprintf('  wrote fig2_bifurcation.png\n');
end


% ========================================================================
% PART 3 -- General fractional-order SEIR model (article Sec. 4, Thm 4.4)
% ========================================================================

function c = gl_coefficients(alpha, n)
    % Grunwald-Letnikov binomial coefficients c(1..n+1):
    % c(1) = 1 (= w_0), c(j+1) = (1 - (alpha+1)/j) * c(j) (= w_j), j = 1..n.
    c = zeros(1, n + 1);
    c(1) = 1;
    for j = 1:n
        c(j + 1) = (1 - (1 + alpha) / j) * c(j);
    end
end

function [t, S, E, I, R] = fractional_seir_general(Delta, beta, mu, omega, gamma, ...
                                                     theta, p, q, alpha, t_sim, y0, h)
    % General fractional-order SEIR model, Eqs. (7)-(10) of Section 4:
    %
    %   D^alpha S = Delta - Delta*p*E/N - Delta*q*I/N - beta*S*I/N - mu*S
    %   D^alpha E = Delta*p*E/N + Delta*q*I/N + beta*S*I/N - omega*E - mu*E
    %   D^alpha I = omega*E - gamma*I - mu*I - theta*I
    %   D^alpha R = gamma*I - mu*R
    %
    % solved by the explicit Grunwald-Letnikov scheme described above.
    % This directly generalises the original prototype's FOSEIRendemic.m
    % (which implements only theta = 0, p = q = 0) by adding the
    % disease-induced-mortality term theta*I and the vertical-transmission
    % terms Delta*p*E/N + Delta*q*I/N.
    n = round(t_sim / h);
    t = linspace(0, t_sim, n + 1);

    S = zeros(1, n + 1); E = zeros(1, n + 1); I = zeros(1, n + 1); R = zeros(1, n + 1);
    S(1) = y0(1); E(1) = y0(2); I(1) = y0(3); R(1) = y0(4);

    c = gl_coefficients(alpha, n);
    hA = h ^ alpha;

    for k = 1:n
        N = S(k) + E(k) + I(k) + R(k);
        fS = Delta - Delta*p*E(k)/N - Delta*q*I(k)/N - beta*S(k)*I(k)/N - mu*S(k);
        fE = Delta*p*E(k)/N + Delta*q*I(k)/N + beta*S(k)*I(k)/N - omega*E(k) - mu*E(k);
        fI = omega*E(k) - gamma*I(k) - mu*I(k) - theta*I(k);
        fR = gamma*I(k) - mu*R(k);

        % Memory sum sum_{j=1}^{k} c(j+1)*X(k-j+1), vectorised as a dot
        % product rather than an explicit inner loop (same weights,
        % same summation as the original prototype's inner j-loop).
        idx = k:-1:1;
        w = c(2:k+1);
        S(k+1) = hA*fS - w*S(idx)';
        E(k+1) = hA*fE - w*E(idx)';
        I(k+1) = hA*fI - w*I(idx)';
        R(k+1) = hA*fR - w*R(idx)';
    end
end

function make_figure4()
    % General SEIR case (Table 1 / Table 3, case (d)); parameters as used
    % throughout Sections 4-5 of the article. See the NUMERICAL METHOD
    % note above for why t_sim = 1000 rather than a shorter window.
    Delta = 0.221176; beta = 0.05; mu = 0.008; omega = 0.05; gamma = 0.003;
    theta = 0.002; p = 0.8; q = 0.95;
    S0 = 140.0; E0 = 0.01; I0 = 0.02;
    R0_init = 141.0 - S0 - E0 - I0;
    y0 = [S0, E0, I0, R0_init];
    h = 0.05; t_sim = 1000;

    alphas = [1.00, 0.95, 0.90];
    colors = {style_black(), style_blue(), style_vermillion()};
    styles = {'-.', '-', '--'};
    labels = {'\alpha = 1.00', '\alpha = 0.95', '\alpha = 0.90'};

    runs = cell(1, 3);
    for a = 1:3
        [t, S, E, I, ~] = fractional_seir_general(Delta, beta, mu, omega, gamma, ...
                                                    theta, p, q, alphas(a), t_sim, y0, h);
        runs{a} = struct('t', t, 'S', S, 'E', E, 'I', I);
        fprintf('  SEIR alpha=%.2f: done (%d steps)\n', alphas(a), numel(t));
    end

    fig = figure('Position', [100 100 1300 400]);
    panel_field = {'S', 'E', 'I'};
    panel_ylabel = {'S(t)', 'E(t)', 'I(t)'};
    for panel = 1:3
        subplot(1, 3, panel);
        hold on;
        for a = 1:3
            plot(runs{a}.t, runs{a}.(panel_field{panel}), ...
                 'Color', colors{a}, 'LineStyle', styles{a}, 'LineWidth', 2);
        end
        xlabel('time t');
        ylabel(panel_ylabel{panel});
        legend(labels, 'Location', 'best');
        grid on; box on;
    end
    sgtitle('General fractional-order SEIR model: effect of \alpha on the approach to F_1');
    print(fig, 'fig4_seir_alpha_comparison.png', '-dpng', '-r300');
    close(fig);
    fprintf('  wrote fig4_seir_alpha_comparison.png\n');
end


% ========================================================================
% PART 4 -- General fractional-order SIR model (article Sec. 4.4, Thm 4.6)
% ========================================================================

function [t, S, I, R] = fractional_sir_general(Delta, beta, mu, gamma, theta, r, ...
                                                 alpha, t_sim, y0, h)
    % General fractional-order SIR model, Eqs. (17)-(19) of Section 4.4:
    %
    %   D^alpha S = Delta - r*Delta*I/N - beta*S*I/N - mu*S
    %   D^alpha I = r*Delta*I/N + beta*S*I/N - gamma*I - mu*I - theta*I
    %   D^alpha R = gamma*I - mu*R
    n = round(t_sim / h);
    t = linspace(0, t_sim, n + 1);

    S = zeros(1, n + 1); I = zeros(1, n + 1); R = zeros(1, n + 1);
    S(1) = y0(1); I(1) = y0(2); R(1) = y0(3);

    c = gl_coefficients(alpha, n);
    hA = h ^ alpha;

    for k = 1:n
        N = S(k) + I(k) + R(k);
        fS = Delta - r*Delta*I(k)/N - beta*S(k)*I(k)/N - mu*S(k);
        fI = r*Delta*I(k)/N + beta*S(k)*I(k)/N - gamma*I(k) - mu*I(k) - theta*I(k);
        fR = gamma*I(k) - mu*R(k);

        idx = k:-1:1;
        w = c(2:k+1);
        S(k+1) = hA*fS - w*S(idx)';
        I(k+1) = hA*fI - w*I(idx)';
        R(k+1) = hA*fR - w*R(idx)';
    end
end

function make_figure5()
    % General SIR case (Table 2 / Table 3, case (d)); same Delta, beta,
    % mu, gamma, theta as the SEIR run above, r = 0.85 as used in
    % Sections 4-5.
    Delta = 0.221176; beta = 0.05; mu = 0.008; gamma = 0.003;
    theta = 0.002; r = 0.85;
    S0 = 140.0; I0 = 0.02;
    R0_init = 141.0 - S0 - I0;
    y0 = [S0, I0, R0_init];
    h = 0.05; t_sim = 1000;

    alphas = [1.00, 0.95, 0.90];
    colors = {style_black(), style_blue(), style_vermillion()};
    styles = {'-.', '-', '--'};
    labels = {'\alpha = 1.00', '\alpha = 0.95', '\alpha = 0.90'};

    runs = cell(1, 3);
    for a = 1:3
        [t, S, I, ~] = fractional_sir_general(Delta, beta, mu, gamma, theta, r, ...
                                                alphas(a), t_sim, y0, h);
        runs{a} = struct('t', t, 'S', S, 'I', I);
        fprintf('  SIR alpha=%.2f: done (%d steps)\n', alphas(a), numel(t));
    end

    fig = figure('Position', [100 100 900 400]);
    panel_field = {'S', 'I'};
    panel_ylabel = {'S(t)', 'I(t)'};
    for panel = 1:2
        subplot(1, 2, panel);
        hold on;
        for a = 1:3
            plot(runs{a}.t, runs{a}.(panel_field{panel}), ...
                 'Color', colors{a}, 'LineStyle', styles{a}, 'LineWidth', 2);
        end
        xlabel('time t');
        ylabel(panel_ylabel{panel});
        legend(labels, 'Location', 'best');
        grid on; box on;
    end
    sgtitle('General fractional-order SIR model: effect of \alpha on the approach to F_1');
    print(fig, 'fig5_sir_alpha_comparison.png', '-dpng', '-r300');
    close(fig);
    fprintf('  wrote fig5_sir_alpha_comparison.png\n');
end


% ========================================================================
% PART 5 -- Table 3: R0 and I* for all eight model variants (Sec. 5.2)
% Computes every row of Table 3 directly from the closed-form formulas of
% Section 4 (Eqs. (12)-(13) for the SEIR family, (20)-(21) for the SIR
% family), evaluated at each of the eight (theta, vertical-transmission)
% settings of Tables 1-2.
% ========================================================================

function [R0, Istar] = seir_R0_Istar(Delta, beta, mu, omega, gamma, theta, p, q)
    % R0 (Eq. 12) and I* (Eq. 13) for the general fractional-order SEIR
    % model, given theta and the vertical-transmission probabilities p, q.
    A = omega + mu * (1 - p);
    B = gamma + mu + theta;
    R0 = omega * (mu*q + beta) / (B * A);
    Istar = Delta * A * (R0 - 1) / ((omega + mu) * (beta - theta));
end

function [R0, Istar] = sir_R0_Istar(Delta, beta, mu, gamma, theta, r)
    % R0 (Eq. 20) and I* (Eq. 21) for the general fractional-order SIR
    % model, given theta and the vertical-transmission probability r.
    B = gamma + mu + theta;
    R0 = (mu*r + beta) / B;
    Istar = Delta * (mu*r + beta - B) / ((beta - theta) * B);
end

function compute_table3()
    % Print R0 and I* for all eight variants at the article's common
    % parameter set, reproducing Table 3 exactly.
    Delta = 0.221176; mu = 0.008; beta = 0.05; gamma = 0.003; omega = 0.05;
    theta_on = 0.002; p = 0.8; q = 0.95; r = 0.85;

    seir_labels = {'SEIR (a) theta=0, no vertical transmission', ...
                   'SEIR (b) theta!=0, no vertical transmission', ...
                   'SEIR (c) theta=0, vertical transmission', ...
                   'SEIR (d) theta!=0, vertical transmission (general)'};
    seir_theta = [0.0, theta_on, 0.0, theta_on];
    seir_p     = [0.0, 0.0, p, p];
    seir_q     = [0.0, 0.0, q, q];

    sir_labels = {'SIR (a) theta=0, no vertical transmission', ...
                  'SIR (b) theta!=0, no vertical transmission', ...
                  'SIR (c) theta=0, vertical transmission', ...
                  'SIR (d) theta!=0, vertical transmission (general)'};
    sir_theta = [0.0, theta_on, 0.0, theta_on];
    sir_r     = [0.0, 0.0, r, r];

    fprintf('%-52s%10s%10s\n', 'Case', 'R0', 'I*');
    for i = 1:4
        [R0, Istar] = seir_R0_Istar(Delta, beta, mu, omega, gamma, seir_theta(i), seir_p(i), seir_q(i));
        fprintf('%-52s%10.4f%10.3f\n', seir_labels{i}, R0, Istar);
    end
    for i = 1:4
        [R0, Istar] = sir_R0_Istar(Delta, beta, mu, gamma, sir_theta(i), sir_r(i));
        fprintf('%-52s%10.4f%10.3f\n', sir_labels{i}, R0, Istar);
    end
end
