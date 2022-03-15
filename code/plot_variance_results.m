function [] = plot_variance_results(total_var_expl, var_expl, add_var_expl, model_names, output_folder)

plot_variance_result(add_var_expl, model_names);
save_figure_kmj([output_folder '/unique_variance']);

plot_variance_result(var_expl, model_names);
save_figure_kmj([output_folder '/variance']);

plot_variance_result(total_var_expl, {'total variance'});
save_figure_kmj([output_folder '/total_variance']);

end

