'use strict';

import {
    Div,
    Form
} from '@src/classes';

export class InlineInputsContainer extends Div {

    constructor(form: Form) {

        super('inline-inputs-container', form);
    }
}