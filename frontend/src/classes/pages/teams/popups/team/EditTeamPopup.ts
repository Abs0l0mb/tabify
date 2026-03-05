'use strict';

import {
    Api,
    Popup,
    Form,
    FormField,
    TextInput,
    ClientLocation
} from '@src/classes';

export class EditTeamPopup extends Popup {

    private form: Form;

    private name: FormField;

    constructor(private teamId: number) {

        super({
            validText: 'Update',
            cancellable: true,
            title: `Edit team ${teamId}`,
        });

        this.build();
    }

    /*
    **
    **
    */
    private async build() : Promise<void> {

        this.form = new Form(this.content);

        //====
        //NAME
        //====

        this.name = this.form.add(new TextInput({
            label: 'Name',
            class: 'mandatory'
        }));

        this.name.linkToErrorKey('name');

        this.populate();
    }

    /*
    **
    **
    */
    private async populate() : Promise<void> {
        
        try {

            let data: any = await Api.get('/team', {
                id: this.teamId
            });

            this.name.input.setValue(data.name);

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

            await Api.post('/team/update', {
                id: this.teamId,
                name: this.name.input.getValue()
            });
            
            this.emit('success');
            this.hide();

            ClientLocation.get().navigation.refresh();
                        
        } catch(error: any) {

            this.form.displayError(error);
            this.validButton.unload();
        }
    }
}