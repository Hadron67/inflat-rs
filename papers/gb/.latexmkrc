$out_dir = 'build';

$aux_dir = 'build/aux';

ensure_path($out_dir);

$pdf_mode = 1;
$pdflatex = 'pdflatex -shell-escape -synctex=1 -interaction=nonstopmode';
@default_files = (
    'main.tex'
);
