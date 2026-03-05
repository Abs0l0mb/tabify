'use strict';

import {
    Popup,
    Api,
    Form,
    FormField,
    TextInput,
    PasswordInput
} from '@src/classes';

export class EditMyDataPopup extends Popup {

    private form: Form;

    private email: FormField;
    private lastName: FormField;
    private firstName: FormField;
    private password: FormField;

    constructor() {

        super({
            validText: 'Update',
            cancellable: true,
            title: `Edit my data`,
        });

        this.build();
    }

    /*
    **
    **
    */
    private async build() : Promise<void> {

        this.form = new Form(this.content);

        //=====
        //EMAIL
        //=====

        this.email = this.form.add(new TextInput({
            label: 'Email',
            mandatory: true
        }));

        this.email.linkToErrorKey('email');

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
        
        //========
        //PASSWORD
        //========

        this.password = this.form.add(new PasswordInput({
            label: 'Password'
        }));

        this.password.linkToErrorKey('password');

        this.populate();
    }

    /*
    **
    **
    */
    private async populate() : Promise<void> {
        
        try {

            let data: any = await Api.get('/me');

            this.email.input.setValue(data.email);
            this.lastName.input.setValue(data.lastName);
            this.firstName.input.setValue(data.firstName);
            
            this.ready();
            
        } catch(error: any) {
            console.log(error);
        }
    }

    /*
    **
    **
    */
    public async onValid() : Promise<void> {

        this.validButton.load();

        try {

            await Api.post('/me/update', {
                email: this.email.input.getValue(),
                firstName: this.email.input.getValue(),
                lastName: this.email.input.getValue(),
                password: this.password.input.getValue()
            });

            this.emit('success', {
                'Email': this.email.input.getValue(),
                'Last Name': this.lastName.input.getValue(),
                'First Name': this.firstName.input.getValue()
            });
            
            this.hide();

        } catch(error: any) {
            
            this.form.displayError(error);
            this.validButton.unload();
        }
    }
}