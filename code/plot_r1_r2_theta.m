function [] = plot_r1_r2_theta(model,bfm_angle,r1,r2)
    
    model_norm = model./max(model);
    angle_vals = unique(bfm_angle);
    rad_vals = unique(r1);

    for i = 1:length(angle_vals)
        subplot(2,4,i)
        idx = find(bfm_angle==angle_vals(i));

        this_angle_data = model_norm(idx);
        this_angle_r1 = r1(idx);
        this_angle_r2 = r2(idx);

        % restructure data into an 8x8 matrix
        unique_rs = unique(r1);
        slice_data = zeros([length(unique(r1)),length(unique(r1))]);
        for sr = 1:length(unique_rs)
            for lr = 1:length(unique_rs)
                valid_idx = find(((this_angle_r1==unique_rs(sr)) & (this_angle_r2==unique_rs(lr))) | ((this_angle_r2==unique_rs(sr)) & (this_angle_r1==unique_rs(lr))));
                valid_data = this_angle_data(valid_idx);
                smaller_r = min(unique_rs(sr),unique_rs(lr));
                larger_r = max(unique_rs(sr),unique_rs(lr));
                x_idx = find(unique_rs==smaller_r);
                y_idx = find(unique_rs==larger_r);
                slice_data(x_idx,y_idx) = mean(valid_data);
            end
        end

        colormap bone
        hBars = bar3(slice_data)
        axis square
        grid on
        set(gcf,'color','w')
        ax = gca;
        ax.FontSize = 10; 
        yticklabels({'0','.','.','.','.','.','.','40'}) % (round(unique_rs,0))
        ytickangle(0)
        xticklabels({'0','.','.','.','.','.','.','40'})
        xtickangle(0)
        zlim([0,1])
        view(-110,30)

        % remove empty bars
        for iSeries = 1:numel(hBars)
            zData = get(hBars(iSeries), 'ZData');  % Get the z data
            index = logical(kron(zData(2:6:end, 2) == 0, ones(6, 1)));  % Find empty bars
            zData(index, :) = nan;                 % Set the z data for empty bars to nan
            set(hBars(iSeries), 'ZData', zData);   % Update the graphics objects
        end
        xlim([0,8.5])
        ylim([0,8.5])

        title(strcat('\theta=',num2str(round(acosd(1-angle_vals(i)))),char(176)), 'fontsize', 10)
    end
    set(gcf,'Position',[100 100 800 400]);
    set(gcf,'renderer','Painters')
    % optionally save figure as vector file
    % print('-dpdf','myVectorFile3.pdf','-r300','-painters')
end

