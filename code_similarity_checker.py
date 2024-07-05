"""
Source code for a plagiarism checker. This script checks submissions
from users against existing data in the database.

Steps in class CodeFingerPrint are based on:
https://theory.stanford.edu/~aiken/publications/papers/sigmod03.pdf
"""

from src.plagiarism.helpers import simplify_code, escape, get_token_coverage
from src.submissions.models import Submission
from src.challenges.models import Challenge
from collections import defaultdict, namedtuple
from src.verification.email import send_email
from config import db

from flask import Flask
import numpy as np
import threading
import argparse
import time
import mmh3
import os


PlagResults = namedtuple(
    'PlagResults',
    [
        'containment',
        'slices',
        'overlap_1',
        'overlap_2',
        'orig_code_1',
        'orig_code_2',
        'offsets_1',
        'offsets_2',
        'file_1',
        'file_2'
    ],
)


def check_for_plagiarism(submission: Submission, app: Flask) -> str | None:
    """
    For the given submission, check if it is plagiarised from another existing
    submission. If no match could be found, this function will return None,
    otherwise an HTML report is returned.

    Input:
        submission: A 'Submission' database entry should be provided for which
                    a plagiarism check must be performed.
        app       : Current flask application; important!
    Output:
        None if no match could be found (and the submission is most likely not
        plagiarised), else an HTML report is returned as a string.
    """
    with app.app_context():
        # Quick fix (hopefully)
        if submission not in db.session:
            submission = db.session.query(Submission).get(submission.id)

        # Find all other submissions for the same challenge as the newly
        # submitted one.
        rows = Submission.query.filter(
            Submission.challenge_id == submission.challenge_id,
            Submission.id != submission.id,
        ).all()

        # Return early if there are no rows.
        if not rows:
            return None

        challenge: Challenge = submission.challenge
        language: str = challenge.language.lower()

        similarity_checker = CodeSimilarityCheck(k=10, win_size=10)
        # Check every submission.
        for row in rows:
            plag_results: PlagResults = (
                similarity_checker.compare_files(
                    submission.file_path, row.file_path, language
                )
            )

            # A matching submission with a containment score of 0.8 or higher
            # is considered likely to be plagiarised from.
            if plag_results.containment >= 0.8:
                highlighted_code_ref, highlighted_code_cand = \
                    similarity_checker.run(plag_results)
                # Create and return an HTML report.
                return similarity_checker.create_report(plag_results,
                                                        highlighted_code_ref,
                                                        highlighted_code_cand)


class PlagiarismChecker:
    """
    Initializing this class creates a thread that searches for similar
    submissions in the background. If there appears to be a case of plagiarism,
    please email the admin set up. Furthermore, the submission will then be
    automatically deleted. If nothing suspicious is found, the created thread
    is gracefully terminated.
    """

    def __init__(self,
                 submission: Submission,
                 email: str | None = None,
                 app: Flask | None = None,
                 test: bool = False):
        """
        Constructor.

        Input:
            submission: The submission that needs to be checked for plagiarism.
            email     : An email address to which a valid report should be
                        sent. Optional, if not set no email will be sent.
            app       : Optional context, useful while testing.
            test      : If set to True, some sleep calls will be skipped, to
                        prevent unnecessary delays.
        """
        self.submission = submission
        self.email = email
        self.test = test

        # Save the current app.
        from flask import current_app as f_app
        self.app = app if app is not None else f_app._get_current_object()

        self.thread = threading.Thread(target=self.run, args=())
        # Start the execution.
        self.thread.start()

    def run(self):
        """
        Checking for plagiarism is an expensive task and is IO heavy, so it is
        better to run this on a separate thread in the 'background'. Creating
        and closing the thread is done automatically and this method should not
        be called manually.
        """
        # Wait 10 seconds before starting the algorithm, I'm afraid that the
        # frontend will crash if we don't :/ (Don't do this if we are testing!)
        if self.test is False:
            time.sleep(10)

        if (report := check_for_plagiarism(self.submission, self.app)) is None:
            return

        # Only sent an email if an email address has been configured.
        if self.email is not None:
            send_email(report,
                       "KodeGrate: Possible Case of Plagiarism",
                       self.email,
                       True)

        with self.app.app_context():
            submission = db.session.merge(self.submission)
            db.session.delete(submission)
            db.session.commit()

    def join(self):
        """
        It is useful while testing to wait for the 'background' thread to
        finish.
        """
        self.thread.join()


class CodeFingerPrint:
    """
    Class fingerprinting and winnowing a file.
    """

    def __init__(self, k, win_size):
        """
        Initialize the CodeFingerPrint object.

        Input:
            k       : Size of the k-grams, a contiguous sequences of
                      k characters.
            win_size: Window size for the winnowing process.
        """
        self.k = k
        self.win_size = win_size

    def generate_k_grams(self, code):
        """
        Generate the sequence of k-grams and hash derived from the code.

        Input:
            code: The source code.

        Output:
            selected_hashes: The winnowed hashes as an array.
        """
        SEED = 823472708
        k_grams = np.array(
            [
                mmh3.hash(code[i: i + self.k], SEED)
                for i in range(len(code) - self.k + 1)
            ]
        )
        return k_grams

    def winnow(self, hashes, remove_duplicates=True):
        """
        Implementation of the Winnowing algorithm. Used for selecting
        fingerprints from hashes of k-grams. In each window select the min-
        imum hash value. If there is more than one hash with the mini-
        mum value, select the rightmost occurrence. Then save all selected
        hashes as the fingerprints of the document.

        Input:
            hashes           : Hashed sequence of k-grams results.
            remove_duplicates: Boolean indicating if duplicate fingerprints
                               should be removed.

        Output:
            selected_hashes: The selected hashes after winnowing as an array.
        """

        def select_min(window):
            """
            Within each window, select the minimum hash value and return its
            position.

            Input:
                hashes: Hashed sequence of k-grams results.

            Output:
                hash_min: The index of the minimum hash value in the window.
            """
            return np.where(window == np.min(window))[0][-1]

        selected_indices = []

        # Create a sequence of hashes of the k-grams.
        for i in range(len(hashes) - self.win_size + 1):
            window = hashes[i: i + self.win_size]
            # Retain positional information derived from the text.
            hash_min_index = i + select_min(window)
            selected_indices.append(hash_min_index)

        selected_hashes = hashes[selected_indices]

        if remove_duplicates:
            selected_hashes, unique_idx = np.unique(
                selected_hashes, return_index=True)
            selected_indices = np.array(selected_indices)[unique_idx]

        return selected_hashes, selected_indices

    def get_fingerprints(self, code):
        """
        Generate and return fingerprints for the given source code.

        This function generates k-grams from the provided code, applies the
        winnowing algorithm to select fingerprints, and Output both a set of
        unique hashes and a dictionary mapping each hash to its positions in
        the code.

        Input:
            code (str): The source code to be fingerprinted.

        Output:
            tuple: A tuple containing:
                    set:  A set of unique hash values (fingerprints).
                    dict: A dictionary where keys are hash values and values
                          are lists of positions where those hashes appear in
                          the code.
        """
        all_hashes = self.generate_k_grams(code)
        hashes, indices = self.winnow(all_hashes, remove_duplicates=False)

        hash_dict = defaultdict(list)
        # Identical hash values may be detected in multiple locations.
        # All such locations are stored to highlight plagiarized code.
        for hash_val, i in zip(hashes, indices):
            hash_dict[hash_val].append(i)

        return set(hashes), dict(hash_dict)

    def get_fingerprints_overlap(self, hashes_1, hashes_2,
                                 indices_1, indices_2):
        """
        Identify overlapping fingerprints between two sets of hashes
        and their indices.

        This function takes two sets of hashes and their corresponding indices,
        finds the intersection of these hashes, and Output the positions
        where these overlapping hashes occur in both sets.

        Input:
            hashes(set)   : The first set of hash values (fingerprints).
            indices (dict): A dictionary mapping each hash to its positions.

        Output:
            tuple: A tuple containing:
                    numpy.ndarray: An array of positions where the overlapping
                                   hashes occur in the first set.
                    numpy.ndarray: An array of positions where the overlapping
                                   hashes occur in the second set.
                    If there is no overlap, both arrays will be empty.
        """
        intersection = hashes_1.intersection(hashes_2)
        if len(intersection) > 0:
            # Find overlap in both. Could be used to mark both reference
            # and candiate file.
            overlap_1 = np.concatenate(
                [np.array(indices_1[i]) for i in intersection])
            overlap_2 = np.concatenate(
                [np.array(indices_2[i]) for i in intersection])
            return overlap_1.flatten(), overlap_2.flatten()
        else:
            return np.array([], dtype=int), np.array([], dtype=int)

    def get_copied_slices(self, indices):
        """
        Identify slices of copied code based on indices.

        This function takes a list of indices, sorts them, and identifies
        contiguous sequences (slices) of indices where copied code appears.
        It Output the start and end indices of these slices.

        Input:
            indices (numpy.ndarray): A list or array of indices indicating
                                     positions of copied code.

        Output:
            numpy.ndarray: A 2D array where the first row contains the start
                           indices and the second row contains the end indices
                           of the contiguous copied slices. If no indices are
                           provided, an empty array is returned.
        """
        if len(indices) == 0:
            return np.array([], [])

        sorted_indices = np.sort(indices)
        # Size of 'next_indices' and 'sorted_indices' should match to compute
        # the gaps.
        next_indices = np.concatenate([sorted_indices[1:], [0]])
        # Find outliers, each token is at most 'k' long. Locations of outliers
        # are used find next start and end positions.
        gaps = np.where(next_indices - sorted_indices > self.k - 1)[0]
        # End positions are calcuated by increasing 'gap' locations with
        # 'k', since each index was originally created in a k-grams window.
        slices_start = np.concatenate(
            [[sorted_indices[0]], sorted_indices[gaps + 1]])
        slices_end = np.concatenate(
            [sorted_indices[gaps] + self.k, [sorted_indices[-1] + self.k]]
        )

        return np.array([slices_start, slices_end])

    def compute_containment(self, slices, indices, code):
        """
        Compute the containment value for a given code based on identified
        slices and indices.

        This function calculates the containment value, which is a measure of
        how much of the code is plagiarized. It uses the identified slices of
        copied code and the indices of tokens in the code to determine the
        token coverage and the overlap of plagiarized tokens.

        Input:
            slices (numpy.ndarray): A 2D array where the first row contain
                                    the start indices and the second row
                                    contains the end indices of the contiguous
                                    copied slices.
            indices (list)        : A list of indices indicating
                                             positions of copied code.
            code (str)            : The source code being analyzed.

        Output:
            float: The containment value, which is the ratio of the token
                   coverage of the plagiarized code to the total token
                   coverage of the file.
        """
        file_token_len = get_token_coverage(self.k, len(code), indices)
        # Calculate the overlap token coverage by summing the lengths
        # of all copied slices.
        overlap_token_len = sum(slices[1] - slices[0])
        # Containment value is calculated by dividing the overlap token
        # coverage by the total token coverage.
        return overlap_token_len / file_token_len


class CodeSimilarityCheck:
    """
    Class checking similarity between code using functionality from Class
    FingerPrinting.
    """

    def __init__(self, k=10, win_size=10):
        """
        Initialize the CodeSimilarityCheck class.

        Input:
            k (int)       : Size of k-grams (sequences of k tokens) for code
                            fingerprinting.
            win_size (int): Size of the window used to extract k-grams from
                            code snippets.
        """
        self.k = k
        self.win_size = win_size
        self.fingerprinter = CodeFingerPrint(k, win_size)

    def preprocess_code(self, code, lang):
        """
        Preprocess the code by simplifying and removing comments based on the
        programming language, and replace variable names with v, function names
        with f, object names with o, and strings with s.

        Input:
            code (str): The source code.
            lang (str): Programming language of the code ('python' or 'c').

        Output:
            code (str)             : The preprocessed code.
            offsets (numpy.ndarray): The offset map to map parts of the
                                     simplified code to the original code.
        """
        code, offsets = simplify_code(code, lang)

        return code, offsets

    def highlight_plag_code(self, slices, offsets, orig_code, color):
        """
        Highlight plagiarized code using the offsets.

        This function takes identified slices of plagiarized code, offsets to
        map the preprocessed code back to the original code, and the original
        code itself. It then returns the original code with highlighted
        plagiarized sections in a specified color.

        Input:
            slices (numpy.ndarray) : A 2D array where the first row contains
                                    the start indices and the second row
                                    contains the end indices of the contiguous
                                    copied slices.
            offsets (numpy.ndarray): A 2D array indicating the start and end
                                     positions of the original code in the
                                     preprocessed code.
            orig_code (str)        : The original code to be highlighted.
            color (str)            : The color to use for highlighting the
                                     plagiarized sections.

        Output:
            highlighted_code (str): The original code with highlighted p
                                    lagiarized sections.
        """
        # The final result is dynamically build over time.
        highlighted_code = "<p>"

        # To skip 'plag' indices, we should remember the end of transformed
        # token and skip every 'plag' instance until it is bigger than
        # 'offset_mem_end'.
        offset_mem_end = -1
        end_last_offset = 0

        for start_idx, end_idx in slices.T:
            # Find the corresponding location in the original code where the
            # plagiarized token starts and ends.
            index_res = offsets[
                (offsets[:, 2] >= start_idx) & (offsets[:, 3] <= end_idx)
            ]
            # Skip if 'plag_idx' can not be translated to a proper token.
            if index_res.size == 0:
                continue

            # Skip the plagiarized tokens from the transformed code by
            # skipping until a value bigger than 'offset_mem_end'.
            if end_idx <= offset_mem_end:
                continue

            for [
                start_offset_orig_code,
                end_offset_orig_code,
                _,
                end_offset_trans_code,
            ] in index_res:

                # Update the offset memory, since we use this index.
                offset_mem_end = end_offset_trans_code

                # Highlight the plagiarized section in red
                highlighted_code += (
                    escape(orig_code[end_last_offset:start_offset_orig_code])
                    + f'<span style="color:{color};">'
                    + escape(
                        orig_code[
                            start_offset_orig_code:end_offset_orig_code+1
                        ]
                    )
                    + "</span>"
                )

                # We only want to copy the part from the end of the previous
                # token up until the current one.
                end_last_offset = end_offset_orig_code + 1

        # Add the last part of the original string if necessary.
        highlighted_code += escape(orig_code[end_last_offset:])
        highlighted_code += "</p>"

        return highlighted_code

    def compare_files(self, file1: str, file2: str, lang: str) -> PlagResults:
        """
        Compare two files and compute the containment value.

        This function reads two files, preprocesses them according to the
        specified programming language, computes their fingerprints, and
        calculates the containment value which represents the similarity ratio
        between the two files.

        Input:
            file1 (str): Path to the reference file.
            file2 (str): Path to the candidate file.
            lang (str) : The programming language used for both files.

        Output:
            PlagResults: An object containing the containment value between
                         the two files, representing the similarity ratio, and
                         other relevant data.
        """
        with open(file1, "r", encoding="utf-8") as f1, open(
            file2, "r", encoding="utf-8"
        ) as f2:
            # Read the content of the files as a single string.
            orig_code_1 = f1.read()
            orig_code_2 = f2.read()

        # Since the 'file1' as the file we are comparing to the other 'file2',
        # we need to remember the offsets of the first file, but not the second
        # one.
        code_1, offsets_1 = self.preprocess_code(orig_code_1, lang)
        code_2, offsets_2 = self.preprocess_code(orig_code_2, lang)

        # Compute the hashes and their position of both.
        hashes_1, indices_1 = self.fingerprinter.get_fingerprints(code_1)
        hashes_2, indices_2 = self.fingerprinter.get_fingerprints(code_2)

        # Hashes of both files should be compared against each other.
        overlap_1, overlap_2 = self.fingerprinter.get_fingerprints_overlap(
            hashes_1, hashes_2, indices_1, indices_2
        )

        # Retrieve start and end positions of plagiarized code.
        slices = self.fingerprinter.get_copied_slices(overlap_1)

        # Compute the containment with respect to 'code_1'.
        containment = self.fingerprinter.compute_containment(
            slices, indices_1, code_1)

        # Create a results object for clear access to individual fields.
        results = PlagResults(containment=containment, slices=slices,
                              overlap_1=overlap_1, overlap_2=overlap_2,
                              orig_code_1=orig_code_1, orig_code_2=orig_code_2,
                              offsets_1=offsets_1, offsets_2=offsets_2,
                              file_1=file1, file_2=file2)

        return results

    def run(self, plag_results: PlagResults):
        """
        Run the similarity check between two specified files.

        This function takes the results from the plagiarism detection process,
        highlights the plagiarized sections in both the reference and
        candidate files, and returns the highlighted code.

        Input:
            plag_results (PlagResults): An object containing the results of
                                        the plagiarism detection process,
                                        including the slices of copied code,
                                        the original code, and the offsets.

        Output:
            tuple: A tuple containing:
                    str: The reference file's code with highlighted plagiarized
                         sections.
                    str: The candidate file's code with highlighted plagiarized
                         sections.
        """
        slices = plag_results.slices
        orig_code_1 = plag_results.orig_code_1
        orig_code_2 = plag_results.orig_code_2
        offsets_1 = plag_results.offsets_1
        offsets_2 = plag_results.offsets_2

        # Highlight plagiarized code reference file.
        highlighted_code_ref = self.highlight_plag_code(
            slices, offsets_1, orig_code_1, "Green")

        # Highlight plagiarized code in candidate file.
        highlighted_code_cand = self.highlight_plag_code(
            slices, offsets_2, orig_code_2, "Red")

        # Uncomment for local testing.
        self.create_report(plag_results, highlighted_code_ref,
                           highlighted_code_cand)

        return highlighted_code_ref, highlighted_code_cand

    def create_report(self, plag_results: PlagResults,
                      highlighted_code_ref, highlighted_code_cand) -> str:
        """
        Create an HTML report based on the plagiarism results and highlighted
        code.

        This function generates an HTML report that includes the plagiarism
        containment score, the original code with highlighted plagiarized
        sections, and writes the report to a file.

        Input:
            plag_results (PlagResults) : A named tuple containing the
                                         plagiarism results.
            highlighted_code_ref (str) : The reference file's code with
                                         highlighted plagiarized sections.
            highlighted_code_cand (str): The candidate file's code with
                                         highlighted plagiarized sections.

        Output:
            str: The generated HTML report.
        """
        containment = plag_results.containment
        file_1 = plag_results.file_1
        file_2 = plag_results.file_2

        # Make sure to load the CSS and prepare for an inline inclusion inside
        # the HTML.
        with open(
                os.path.dirname(os.path.realpath(__file__)) + '/style.css'
        ) as css_file:
            css = css_file.read()

        html_report = (
            '<!DOCTYPE html>\n'
            '<html lang="en">\n'
            '<head>\n'
            '    <meta charset="UTF-8">\n'
            '    <meta name="viewport" content="width=device-width, '
            'initial-scale=1.0">\n'
            '    <style>\n'
            f"{css}"
            '    </style>\n'
            '    <title>Plagiarism Report</title>\n'
            '</head>\n'
            '<body>\n'
            '    <h>Reference file: <span class="reference-file">'
            f"{file_1}</span></h>\n"
            '    <h>Plagiarized file: <span class="plagiarized-file">'
            f"{file_2}</span></h>\n"
            '    <h2>Containment: <span class="containment">'
            f"{containment:.2f}</span></h2>\n"
            '    <div class="code-container">\n'
            '        <div class="code-block">\n'
            '            <h2>Reference code</h2>\n'
            '            <pre><code class="code-original">'
            f"{highlighted_code_ref}</code></pre>\n"
            '        </div>\n'
            '        <div class="code-block">\n'
            '            <h2>Plagiarized code</h2>\n'
            '            <pre><code class="code-plagiarized">'
            f"{highlighted_code_cand}</code></pre>\n"
            '        </div>\n'
            '    </div>\n'
            '</body>\n'
            '</html>'
        )

        with open("report.html", "w") as file:
            file.write(html_report)

        return html_report


def main():
    parser = argparse.ArgumentParser(
        description="Check code similarity between two files."
    )
    parser.add_argument("file1", type=str, help="The reference file")
    parser.add_argument("file2", type=str, help="The candidate file")
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="The programming language of the files (python or c)",
        choices=["python", "c"],
    )
    parser.add_argument(
        "--k", type=int, default=10, help="The k-gram length (default: 10)"
    )
    parser.add_argument(
        "--win_size", type=int, default=10,
        help="The window size (default: 10)"
    )

    args = parser.parse_args()

    similarity_checker = CodeSimilarityCheck(k=args.k, win_size=args.win_size)
    plag_results: PlagResults = similarity_checker.compare_files(
        args.file1, args.file2, lang=args.lang)
    similarity_checker.run(plag_results)


if __name__ == "__main__":
    main()
