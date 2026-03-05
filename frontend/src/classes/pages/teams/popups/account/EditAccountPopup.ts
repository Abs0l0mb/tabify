'use strict';

import {
    Div, 
    Api,
    Popup,
    Form,
    FormField,
    TextInput,
    SelectInput,
    PasswordInput,
    Checkbox
} from '@src/classes';

export interface AccessRight {
    id: number,
    checkbox: Checkbox
}

export class EditAccountPopup extends Popup {

    private form: Form;

    private lastName: FormField;
    private firstName: FormField;
    private email: FormField;
    private password: FormField;
    private team: FormField;
    private accessRights: AccessRight[] = [];

    constructor(private accountId: number) {

        super({
            validText: 'Update',
            cancellable: true,
            title: `Edit account ${accountId}`,
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
            mandatory: false
        }));

        this.password.linkToErrorKey('password');
        
        //====
        //TEAM
        //====

        this.team = this.form.add(new SelectInput({
            label: 'Team',
            items: await this.getTeams(),
            mandatory: true
        }));

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

        this.populate();
    }

    /*
    **
    **
    */
    private async populate() : Promise<void> {
        
        try {

            let data: any = await Api.get('/account', {
                id: this.accountId
            });

            this.email.input.setValue(data.email);
            this.lastName.input.setValue(data.last_name);
            this.firstName.input.setValue(data.first_name);
            this.team.input.setValue(data.team_id);

            if (data.access_rights) {
                
                for (let accessRightId of data.access_rights) {
                    checkbox_loop:
                    for (let accessRight of this.accessRights) {
                        if (accessRight.id === parseInt(accessRightId)) {
                            accessRight.checkbox.setValue(true);
                            break checkbox_loop;
                        }
                    }
                }
            }

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

            const password = this.password.input.getValue();

            await Api.post('/account/update', {
                id: this.accountId,
                teamId: this.team.input.getValue(),
                firstName: this.firstName.input.getValue(),
                lastName: this.lastName.input.getValue(),
                email: this.email.input.getValue(),
                password: password ? password : null,
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

    /*
    **
    **
    */
    private async getTeams() : Promise<any[]> {
        
        let output : any[] = [];

        let data = await Api.get('/teams');

        for (let row of data) {
            output.push({
                label: row.name,
                value: row.id
            });
        }

        return output;
    }
}