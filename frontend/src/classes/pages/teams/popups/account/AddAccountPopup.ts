'use strict';

import {
    Popup,
    Div, 
    Api,
    Form,
    FormField,
    TextInput,
    PasswordInput,
    Checkbox
} from '@src/classes';

export interface AccessRight {
    id: number,
    checkbox: Checkbox
}

export class AddAccountPopup extends Popup {

    private form: Form;

    private lastName: FormField;
    private firstName: FormField;
    private email: FormField;
    private password: FormField;
    private accessRights: AccessRight[] = [];

    constructor(private teamId: number) {

        super({
            validText: 'Add',
            cancellable: true,
            title: `Add account`,
        });

        this.build();
    }

    /*
    **
    **
    */
    private async build() : Promise<void> {

        this.form = new Form(this.content);

        //=========
        //LAST NAME
        //=========

        this.lastName = this.form.add(new TextInput({
            label: 'Last name',
            mandatory: true
        }));

        this.lastName.linkToErrorKey('lastName');

        //==========
        //FIRST NAME
        //==========

        this.firstName = this.form.add(new TextInput({
            label: 'First name',
            mandatory: true
        }));

        this.firstName.linkToErrorKey('firstName');

        //=====
        //EMAIL
        //=====

        this.email = this.form.add(new TextInput({
            label: 'Email',
            mandatory: true
        }));

        this.email.linkToErrorKey('email');

        //========
        //PASSWORD
        //========

        this.password = this.form.add(new PasswordInput({
            label: 'Password',
            mandatory: true
        }));

        this.password.linkToErrorKey('password');

        //======
        //RIGHTS
        //======

        new Div('checkboxes-label', this.form).write('Access rights');

        for (let accessRight of await Api.get('/access-rights')) {

            const checkbox = new Checkbox({
                label: accessRight.name
            });

            this.accessRights.push({
                id: accessRight.id,
                checkbox: checkbox
            });

            this.form.add(checkbox);
        }

        this.ready();
    }

    /*
    **
    **
    */
    public async onValid() : Promise<void> {

        this.validButton.load();

        try {

            await Api.post('/team/accounts/add', {
                teamId: this.teamId,
                lastName: this.lastName.input.getValue(),
                firstName: this.firstName.input.getValue(),
                email: this.email.input.getValue(),
                password: this.password.input.getValue(),
                accessRights: this.getCheckedAccessRights()
            });

            this.emit('success');
            this.hide();
                        
        } catch(error: any) {

            this.form.displayError(error);
            this.validButton.unload();
        }
    }

    /*
    **
    **
    */
    private getCheckedAccessRights() : number[] {

        const output: number[] = [];

        for (let accessRight of this.accessRights) {
            if (accessRight.checkbox.getValue())
                output.push(accessRight.id);
        }

        return output;
    }
}