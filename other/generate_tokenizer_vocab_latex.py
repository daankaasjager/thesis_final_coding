import json
import argparse

def generate_latex_rows(vocab_dict):
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    rows = []
    for token, idx in sorted_vocab:
        token_escaped = token.replace('_', r'\_')  # escape underscores
        rows.append(f"{idx} & {token_escaped} \\\\")
    return "\n".join(rows)

def main(atom_pair_file, motif_file, output_file):
    with open(atom_pair_file, 'r') as f:
        atom_pair_vocab = json.load(f)["model"]["vocab"]
    with open(motif_file, 'r') as f:
        motif_vocab = json.load(f)["model"]["vocab"]

    atom_pair_latex = generate_latex_rows(atom_pair_vocab)
    motif_latex = generate_latex_rows(motif_vocab)

    with open(output_file, 'w') as f:
        f.write(r"""\section*{Tokenizer Vocabularies}
\addcontentsline{toc}{section}{Tokenizer Vocabularies}

\subsection*{A.1 Atom-Pair Encoding Tokenizer Vocabulary}
\begin{longtable}{rl}
\toprule
\textbf{ID} & \textbf{Token} \\
\midrule
""")
        f.write(atom_pair_latex)
        f.write(r"""
\bottomrule
\end{longtable}

\subsection*{A.2 WordLevel Tokenizer with Motifs Vocabulary}
\begin{longtable}{rl}
\toprule
\textbf{ID} & \textbf{Token} \\
\midrule
""")
        f.write(motif_latex)
        f.write(r"""
\bottomrule
\end{longtable}
""")

    print(f"LaTeX table written to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--atom_pair_file", required=True, help="Path to Atom-Pair Encoding tokenizer JSON")
    parser.add_argument("--motif_file", required=True, help="Path to Motif tokenizer JSON")
    parser.add_argument("--output_file", default="tokenizer_vocab_tables.tex", help="Output LaTeX file")
    args = parser.parse_args()

    main(args.atom_pair_file, args.motif_file, args.output_file)
