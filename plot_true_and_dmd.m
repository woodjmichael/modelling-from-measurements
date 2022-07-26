function plot_true_and_dmd(x,y,xhat,yhat,title_name)
    rmse = sqrt(mean(mean([(x - xhat ).^2, ( y - yhat ).^2])));

    title_name = sprintf('%s %s %.2f)',title_name, '(RMSE', rmse );
    t = 1845:2:1903;
    t = t(1:length(x));
    figure()
    plot(t, x, 'b')
    hold on
    plot(t, xhat,'b--')
    hold on
    plot(t, y, 'r')
    hold on
    plot(t, yhat, 'r--')
    title(title_name)
    legend('Prey true', 'Prey model', 'Predator true', 'Predator model')