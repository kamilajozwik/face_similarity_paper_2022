function [ ] = write_as_tsv( filename, data, column_headers, row_headers )
    fid = fopen(filename, 'w') ;
    if fid == -1
        error(['Could not open file for writing (could be missing directory?): ' filename]);
    end
    
    if ~iscell(data)
        data = num2cell(data);
    end
    
    if nargin == 4
        fprintf(fid, '\t');
    end
    
    [rows, cols] = size(data);
    
    for c = 1:cols-1
        fprintf(fid, '%s\t', column_headers{c});
    end
    fprintf(fid, '%s\n', column_headers{cols}) ;
    
    for r = 1:rows
        if nargin == 4
            fprintf(fid, '%s\t', row_headers{r});
        end
        for c = 1:cols-1
            fprintf(fid, '%s\t', num2str(data{r,c}));
        end
        fprintf(fid, '%s\n', num2str(data{r, end}));        
    end
    
    fclose(fid) ;
end

