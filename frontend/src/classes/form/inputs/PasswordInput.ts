'use strict';

import {
    TextInput,
    Block
} from '@src/classes';

export interface PasswordInputData {
    label: string;
    mandatory?: boolean;
    autocomplete?: boolean;
    class?: string;
}


export class PasswordInput extends TextInput {

    private isPasswordVisible: boolean = false;

    constructor(public data: PasswordInputData, parent?: Block) {

        super(data, parent);

        this.setCustomType('password');
        
        this.input.setAttribute('type', 'password');

        this.customTypeIcon.onNative('mousedown', this.toggleType.bind(this));
    }

    /*
    **
    **
    */
    private toggleType() : void {

        this.isPasswordVisible = !this.isPasswordVisible;

        this.input.setAttribute('type', this.isPasswordVisible ? 'text' : 'password');

        this.setCustomTypeIconActivated(this.isPasswordVisible);
    }
}