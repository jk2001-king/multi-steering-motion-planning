#!/usr/bin/env python3
"""Append-only raw forward PASS audit and second reverse MODEL sensitivity screen."""

from hospital_h3b_post_forward import BASE,run_review


def main():
    number=1
    while (output:=BASE/f'review_r{number:02d}').exists():number+=1
    run_review(output)


if __name__=='__main__':main()
