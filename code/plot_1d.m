function [] = plot_1d( f, caption, color )
    global all_eu_distances
    global all_similarities
    plot(all_eu_distances, all_similarities, 'ro', 'MarkerSize',2);
    plot(1:80, f(1:80),'Color', color, 'LineWidth',4);
    ylim([0 1]);
    xlabel('Euclidean distance');
    ylabel('Similarity judgements');
    title(caption);
end

