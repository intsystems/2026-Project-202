# Build both camera-ready PDFs unless specific targets are requested.
@default_files = qw(artifacts_camera_ready.tex artifacts_camera_ready_codex.tex);

# Keep native Windows BibTeX in the source directory when using MSYS Perl.
$bibtex_fudge = 0 if $^O eq 'msys';
